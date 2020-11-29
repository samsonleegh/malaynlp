# !pip3 install xlnet-tensorflow
import xlnet
import xlnet.modeling, xlnet.xlnet
import transformers

import argparse
import itertools
import os.path
import time
import shutil
import re
import json

import torch
import torch.optim.lr_scheduler
import tensorflow as tf
from tqdm import tqdm
import numpy as np
from tensorflow.tools.graph_transforms import TransformGraph

import evaluate
import trees_newline as trees
import vocabulary
import nkutil
import parse_nk_xlnet_base as parse_nk
import tokens
from parse_nk_xlnet_base import BERT_TOKEN_MAPPING

def run(args):
    import trees_newline as trees
    test_treebank = trees.load_trees(args.test_path)

    info = torch.load(args.model_path)

    parser = parse_nk.NKChartParser.from_spec(info['spec'], info['state_dict'])
    bert_model = info['spec']['hparams']['bert_model']

    tf.compat.v1.reset_default_graph()
    sess = tf.InteractiveSession()
    sd = parser.state_dict()

    LABEL_VOCAB = [x[0] for x in sorted(parser.label_vocab.indices.items(), key=lambda x: x[1])]
    TAG_VOCAB = [x[0] for x in sorted(parser.tag_vocab.indices.items(), key=lambda x: x[1])]

    position_table = tf.constant(sd['embedding.position_table'].cpu().numpy(), name="position_table")


    def make_layer_norm(input, torch_name, name):
        # TODO(nikita): The epsilon here isn't quite the same as in pytorch
        # The pytorch code adds eps=1e-3 to the standard deviation, while this
        # tensorflow code adds eps=1e-6 to the variance.
        # However, the resulting mismatch in floating-point values does not seem to
        # translate to any noticable changes in the parser's tree output
        mean, variance = tf.nn.moments(input, [1], keep_dims=True)
        return tf.nn.batch_normalization(
            input,
            mean, variance,
            offset=tf.constant(sd[f'{torch_name}.b_2'].cpu().numpy(), name=f"{name}/offset"),
            scale=tf.constant(sd[f'{torch_name}.a_2'].cpu().numpy(), name=f"{name}/scale"),
            variance_epsilon=1e-6)

    def make_heads(input, shape_bthf, shape_xtf, torch_name, name):
        res = tf.matmul(input,
            tf.constant(sd[torch_name].cpu().numpy().transpose((1,0,2)).reshape((512, -1)), name=f"{name}/W"))
        res = tf.reshape(res, shape_bthf)
        res = tf.transpose(res, (0,2,1,3)) # batch x num_heads x time x feat
        res = tf.reshape(res, shape_xtf) # _ x time x feat
        return res

    def make_attention(input, nonpad_ids, dim_flat, dim_padded, valid_mask, torch_name, name):
        input_flat = tf.scatter_nd(indices=nonpad_ids[:, None], updates=input, shape=tf.concat([dim_flat, tf.shape(input)[1:]], axis=0))
        input_flat_dat, input_flat_pos = tf.split(input_flat, 2, axis=-1)

        shape_bthf = tf.concat([dim_padded, [8, -1]], axis=0)
        shape_bhtf = tf.convert_to_tensor([dim_padded[0], 8, dim_padded[1], -1])
        shape_xtf = tf.convert_to_tensor([dim_padded[0] * 8, dim_padded[1], -1])
        shape_xf = tf.concat([dim_flat, [-1]], axis=0)

        qs1 = make_heads(input_flat_dat, shape_bthf, shape_xtf, f'{torch_name}.w_qs1', f'{name}/q_dat')
        ks1 = make_heads(input_flat_dat, shape_bthf, shape_xtf, f'{torch_name}.w_ks1', f'{name}/k_dat')
        vs1 = make_heads(input_flat_dat, shape_bthf, shape_xtf, f'{torch_name}.w_vs1', f'{name}/v_dat')
        qs2 = make_heads(input_flat_pos, shape_bthf, shape_xtf, f'{torch_name}.w_qs2', f'{name}/q_pos')
        ks2 = make_heads(input_flat_pos, shape_bthf, shape_xtf, f'{torch_name}.w_ks2', f'{name}/k_pos')
        vs2 = make_heads(input_flat_pos, shape_bthf, shape_xtf, f'{torch_name}.w_vs2', f'{name}/v_pos')

        qs = tf.concat([qs1, qs2], axis=-1)
        ks = tf.concat([ks1, ks2], axis=-1)
        attn_logits = tf.matmul(qs, ks, transpose_b=True) / (1024 ** 0.5)

        attn_mask = tf.reshape(tf.tile(valid_mask, [1,8*dim_padded[1]]), tf.shape(attn_logits))
        # TODO(nikita): use tf.where and -float('inf') here?
        attn_logits -= 1e10 * tf.to_float(~attn_mask)

        attn = tf.nn.softmax(attn_logits)

        attended_dat_raw = tf.matmul(attn, vs1)
        attended_dat_flat = tf.reshape(tf.transpose(tf.reshape(attended_dat_raw, shape_bhtf), (0,2,1,3)), shape_xf)
        attended_dat = tf.gather(attended_dat_flat, nonpad_ids)
        attended_pos_raw = tf.matmul(attn, vs2)
        attended_pos_flat = tf.reshape(tf.transpose(tf.reshape(attended_pos_raw, shape_bhtf), (0,2,1,3)), shape_xf)
        attended_pos = tf.gather(attended_pos_flat, nonpad_ids)

        out_dat = tf.matmul(attended_dat, tf.constant(sd[f'{torch_name}.proj1.weight'].cpu().numpy().transpose()))
        out_pos = tf.matmul(attended_pos, tf.constant(sd[f'{torch_name}.proj2.weight'].cpu().numpy().transpose()))

        out = tf.concat([out_dat, out_pos], -1)
        return make_layer_norm(input + out, f'{torch_name}.layer_norm', f'{name}/layer_norm')

    def make_dense_relu_dense(input, torch_name, torch_type, name):
        # TODO: use name
        mul1 = tf.matmul(input, tf.constant(sd[f'{torch_name}.w_1{torch_type}.weight'].cpu().numpy().transpose()))
        mul1b = tf.nn.bias_add(mul1, tf.constant(sd[f'{torch_name}.w_1{torch_type}.bias'].cpu().numpy()))
        mul1b = tf.nn.relu(mul1b)
        mul2 = tf.matmul(mul1b, tf.constant(sd[f'{torch_name}.w_2{torch_type}.weight'].cpu().numpy().transpose()))
        mul2b = tf.nn.bias_add(mul2, tf.constant(sd[f'{torch_name}.w_2{torch_type}.bias'].cpu().numpy()))
        return mul2b

    def make_ff(input, torch_name, name):
        # TODO: use name
        input_dat, input_pos = tf.split(input, 2, axis=-1)
        out_dat = make_dense_relu_dense(input_dat, torch_name, 'c', name="TODO_dat")
        out_pos = make_dense_relu_dense(input_pos, torch_name, 'p', name="TODO_pos")
        out = tf.concat([out_dat, out_pos], -1)
        return make_layer_norm(input + out, f'{torch_name}.layer_norm', f'{name}/layer_norm')

    def make_stacks(input, nonpad_ids, dim_flat, dim_padded, valid_mask, num_stacks):
        res = input
        for i in range(num_stacks):
            res = make_attention(res, nonpad_ids, dim_flat, dim_padded, valid_mask, f'encoder.attn_{i}', name=f'attn_{i}')
            res = make_ff(res, f'encoder.ff_{i}', name=f'ff_{i}')
        return res

    def make_layer_norm_with_constants(input, constants):
        # TODO(nikita): The epsilon here isn't quite the same as in pytorch
        # The pytorch code adds eps=1e-3 to the standard deviation, while this
        # tensorflow code adds eps=1e-6 to the variance.
        # However, the resulting mismatch in floating-point values does not seem to
        # translate to any noticable changes in the parser's tree output
        mean, variance = tf.nn.moments(input, [1], keep_dims=True)
        return tf.nn.batch_normalization(
            input,
            mean, variance,
            offset=constants[0],
            scale=constants[1],
            variance_epsilon=1e-6)

    def make_flabel_with_constants(input, constants):
        mul1 = tf.matmul(input, constants[0])
        mul1b = tf.nn.bias_add(mul1, constants[1])
        mul1b = make_layer_norm_with_constants(mul1b, constants[2:4])
        mul1b = tf.nn.relu(mul1b)
        mul2 = tf.matmul(mul1b, constants[4])
        mul2b = tf.nn.bias_add(mul2, constants[5], name='flabel')
        return mul2b

    def make_ftag(input):
        constants = (
            tf.constant(sd['f_tag.0.weight'].cpu().numpy().transpose()),
            tf.constant(sd['f_tag.0.bias'].cpu().numpy()),
            tf.constant(sd['f_tag.1.b_2'].cpu().numpy(), name="tag/layer_norm/offset"),
            tf.constant(sd['f_tag.1.a_2'].cpu().numpy(), name="tag/layer_norm/scale"),
            tf.constant(sd['f_tag.3.weight'].cpu().numpy().transpose()),
            tf.constant(sd['f_tag.3.bias'].cpu().numpy()),
        )
        mul1 = tf.matmul(input, constants[0])
        mul1b = tf.nn.bias_add(mul1, constants[1])
        mul1b = make_layer_norm_with_constants(mul1b, constants[2:4])
        mul1b = tf.nn.relu(mul1b)
        mul2 = tf.matmul(mul1b, constants[4])
        mul2b = tf.nn.bias_add(mul2, constants[5], name='ftag')
        return mul2b

    def make_flabel_constants():
        return (
            tf.constant(sd['f_label.0.weight'].cpu().numpy().transpose()),
            tf.constant(sd['f_label.0.bias'].cpu().numpy()),
            tf.constant(sd['f_label.1.b_2'].cpu().numpy(), name="label/layer_norm/offset"),
            tf.constant(sd['f_label.1.a_2'].cpu().numpy(), name="label/layer_norm/scale"),
            tf.constant(sd['f_label.3.weight'].cpu().numpy().transpose()),
            tf.constant(sd['f_label.3.bias'].cpu().numpy()),
        )

    def make_network():
        # batch x num_subwords
        input_ids = tf.placeholder(shape=(None, None), dtype=tf.int32, name='input_ids')
        word_end_mask = tf.placeholder(shape=(None, None), dtype=tf.int32, name='word_end_mask')
        input_dat, nonpad_ids, dim_flat, dim_padded, valid_mask, lengths, bert_features = make_bert(input_ids, word_end_mask)
        input_pos_flat = tf.tile(position_table[:dim_padded[1]], [dim_padded[0], 1])
        input_pos = tf.gather(input_pos_flat, nonpad_ids)

        input_joint = tf.concat([input_dat, input_pos], -1)
        input_joint = make_layer_norm(input_joint, 'embedding.layer_norm', 'embedding/layer_norm')

        word_out = make_stacks(input_joint, nonpad_ids, dim_flat, dim_padded, valid_mask, num_stacks=parser.spec['hparams']['num_layers'])
        word_out = tf.concat([word_out[:, 0::2], word_out[:, 1::2]], -1)

        # part-of-speech predictions
        ftag = make_ftag(word_out)
        tags_packed = tf.argmax(ftag, axis=-1)
        tags = tf.reshape(
            tf.scatter_nd(indices=nonpad_ids[:, None], updates=tags_packed, shape=dim_flat),
            dim_padded
            )
        tags = tf.identity(tags, name="tags")

        fp_out = tf.concat([word_out[:-1,:512], word_out[1:,512:]], -1)

        fp_start_idxs = tf.cumsum(lengths, exclusive=True)
        fp_end_idxs = tf.cumsum(lengths) - 1 # the number of fenceposts is 1 less than the number of words

        fp_end_idxs_uneven = fp_end_idxs - tf.convert_to_tensor([1, 0])

        # Have to make these outside tf.map_fn for model compression to work
        constants = make_flabel_constants()

        def to_map(start_and_end):
            start, end = start_and_end
            fp = fp_out[start:end]
            flabel = make_flabel_with_constants(tf.reshape(fp[None,:,:] - fp[:,None,:], (-1, 1024)), constants)
            actual_chart_size = end-start
            flabel = tf.reshape(flabel, [actual_chart_size, actual_chart_size, -1])
            amount_to_pad = dim_padded[1] - actual_chart_size
            # extra padding on the label dimension is for the not-a-constituent label,
            # which always has a score of 0
            flabel = tf.pad(flabel, [[0, amount_to_pad], [0, amount_to_pad], [1, 0]])
            return flabel

        charts = tf.map_fn(to_map, (fp_start_idxs, fp_end_idxs), dtype=(tf.float32))
        charts = tf.identity(charts, name="charts")

        return input_ids, word_end_mask, charts, tags, bert_features

    def build_tf_xlnet_to_pytorch_map(model, config, tf_weights=None):
        """ A map of modules from TF to PyTorch.
            I use a map to keep the PyTorch model as
            identical to the original PyTorch model as possible.
        """

        tf_to_pt_map = {}

        if hasattr(model, "transformer"):
            if hasattr(model, "lm_loss"):
                # We will load also the output bias
                tf_to_pt_map["model/lm_loss/bias"] = model.lm_loss.bias
            if hasattr(model, "sequence_summary") and "model/sequnece_summary/summary/kernel" in tf_weights:
                # We will load also the sequence summary
                tf_to_pt_map["model/sequnece_summary/summary/kernel"] = model.sequence_summary.summary.weight
                tf_to_pt_map["model/sequnece_summary/summary/bias"] = model.sequence_summary.summary.bias
            if (
                hasattr(model, "logits_proj")
                and config.finetuning_task is not None
                and "model/regression_{}/logit/kernel".format(config.finetuning_task) in tf_weights
            ):
                tf_to_pt_map["model/regression_{}/logit/kernel".format(config.finetuning_task)] = model.logits_proj.weight
                tf_to_pt_map["model/regression_{}/logit/bias".format(config.finetuning_task)] = model.logits_proj.bias

            # Now load the rest of the transformer
            model = model.transformer

        # Embeddings and output
        tf_to_pt_map.update(
            {
                "model/transformer/word_embedding/lookup_table": model.word_embedding.weight,
                "model/transformer/mask_emb/mask_emb": model.mask_emb,
            }
        )

        # Transformer blocks
        for i, b in enumerate(model.layer):
            layer_str = "model/transformer/layer_%d/" % i
            tf_to_pt_map.update(
                {
                    layer_str + "rel_attn/LayerNorm/gamma": b.rel_attn.layer_norm.weight,
                    layer_str + "rel_attn/LayerNorm/beta": b.rel_attn.layer_norm.bias,
                    layer_str + "rel_attn/o/kernel": b.rel_attn.o,
                    layer_str + "rel_attn/q/kernel": b.rel_attn.q,
                    layer_str + "rel_attn/k/kernel": b.rel_attn.k,
                    layer_str + "rel_attn/r/kernel": b.rel_attn.r,
                    layer_str + "rel_attn/v/kernel": b.rel_attn.v,
                    layer_str + "ff/LayerNorm/gamma": b.ff.layer_norm.weight,
                    layer_str + "ff/LayerNorm/beta": b.ff.layer_norm.bias,
                    layer_str + "ff/layer_1/kernel": b.ff.layer_1.weight,
                    layer_str + "ff/layer_1/bias": b.ff.layer_1.bias,
                    layer_str + "ff/layer_2/kernel": b.ff.layer_2.weight,
                    layer_str + "ff/layer_2/bias": b.ff.layer_2.bias,
                }
            )

        # Relative positioning biases
        if config.untie_r:
            r_r_list = []
            r_w_list = []
            r_s_list = []
            seg_embed_list = []
            for b in model.layer:
                r_r_list.append(b.rel_attn.r_r_bias)
                r_w_list.append(b.rel_attn.r_w_bias)
                r_s_list.append(b.rel_attn.r_s_bias)
                seg_embed_list.append(b.rel_attn.seg_embed)
        else:
            r_r_list = [model.r_r_bias]
            r_w_list = [model.r_w_bias]
            r_s_list = [model.r_s_bias]
            seg_embed_list = [model.seg_embed]
        tf_to_pt_map.update(
            {
                "model/transformer/r_r_bias": r_r_list,
                "model/transformer/r_w_bias": r_w_list,
                "model/transformer/r_s_bias": r_s_list,
                "model/transformer/seg_embed": seg_embed_list,
            }
        )
        return tf_to_pt_map

    def make_bert(input_ids, word_end_mask):
        # We can derive input_mask from either input_ids or word_end_mask
        input_mask = (1 - tf.cumprod(1 - word_end_mask, axis=-1, reverse=True))
        input_mask = (1 - input_mask)
        # input_mask = word_end_mask
        token_type_ids = tf.zeros_like(input_ids)
        bert_model = make_bert_instance(input_ids, input_mask, token_type_ids)

        bert_features = bert_model.get_sequence_output()
        print(bert_features)
        bert_features = tf.transpose(bert_features, perm=[1, 0, 2])
        bert_features_packed = tf.gather(
            tf.reshape(bert_features, [-1, int(bert_features.shape[-1])]),
            tf.to_int32(tf.where(tf.reshape(word_end_mask, (-1,))))[:,0])
        projected_annotations = tf.matmul(
            bert_features_packed,
            tf.constant(sd['project_bert.weight'].cpu().numpy().transpose()))

        # input_mask is over subwords, whereas valid_mask is over words
        sentence_lengths = tf.reduce_sum(word_end_mask, -1)
        valid_mask = (tf.range(tf.reduce_max(sentence_lengths))[None,:] < sentence_lengths[:, None])
        dim_padded = tf.shape(valid_mask)[:2]
        mask_flat = tf.reshape(valid_mask, (-1,))
        dim_flat = tf.shape(mask_flat)[:1]
        nonpad_ids = tf.to_int32(tf.where(mask_flat)[:,0])

        return projected_annotations, nonpad_ids, dim_flat, dim_padded, valid_mask, sentence_lengths, bert_features

    def make_bert_instance(input_ids, input_mask, token_type_ids):
        # Transfer BERT config into tensorflow implementation
        import json
        
        input_ids = tf.transpose(input_ids, [1, 0])
        token_type_ids = tf.transpose(token_type_ids, [1, 0])
        input_mask = tf.transpose(input_mask, [1, 0])
        
        c = parser.bert.config.to_dict()
        c["n_token"] = 32000
        with open('export_model/xlnet-config.json', 'w') as fopen:
            json.dump(c, fopen)

        config = xlnet.xlnet.XLNetConfig(json_path = 'export_model/xlnet-config.json')

        kwargs = dict(
            is_training=False,
            use_tpu=False,
            use_bfloat16=False,
            dropout=0,
            dropatt=0,
            init='normal',
            init_range=0.1,
            init_std=0.02,
            clamp_len=-1)
        
        xlnet_parameters = xlnet.xlnet.RunConfig(**kwargs)
        model = xlnet.xlnet.XLNetModel(
            xlnet_config=config,
            run_config=xlnet_parameters,
            input_ids=input_ids,
            seg_ids=token_type_ids,
            input_mask=tf.cast(input_mask, tf.float32))
        
        mapping = build_tf_xlnet_to_pytorch_map(parser.bert, parser.bert.config)
        
        bert_variables = [v for v in tf.get_collection('variables')]
        tf.variables_initializer(bert_variables).run()

        for i in range(len(bert_variables)):
            variable = bert_variables[i]
            name = variable.name.split(':')[0]
            print(name)
            if "kernel" in name and ("ff" in name or "summary" in name or "logit" in name):
                print("Transposing")
                array = mapping[name].T
            elif isinstance(mapping[name], list):
                print(mapping[name][0].shape)
                c = [torch.unsqueeze(t, 0) for t in mapping[name]]
                print(c[0].shape)
                array = torch.cat(c, 0)
            else:
                array = mapping[name]
                
            variable.load(array.detach().cpu().numpy())
        
        return model

    ########### create tf network
    print("creating tf network")
    the_inp_tokens, the_inp_mask, the_out_chart, the_out_tags, bert_features = make_network()

    def bertify_batch(sentences):
        all_input_ids = np.zeros((len(sentences), parser.bert_max_len), dtype=int)
        all_input_mask = np.zeros((len(sentences), parser.bert_max_len), dtype=int)
        all_word_end_mask = np.zeros((len(sentences), parser.bert_max_len), dtype=int)

        subword_max_len = 0
        for snum, sentence in enumerate(sentences):
            tokens = []
            word_end_mask = []

            cleaned_words = []
            for word in sentence:
                word = BERT_TOKEN_MAPPING.get(word, word)
                if word == "n't" and cleaned_words:
                    cleaned_words[-1] = cleaned_words[-1] + "n"
                    word = "'t"
                cleaned_words.append(word)

            for word in cleaned_words:
                word_tokens = parser.bert_tokenizer.tokenize(word)
                for _ in range(len(word_tokens)):
                    word_end_mask.append(0)
                word_end_mask[-1] = 1
                tokens.extend(word_tokens)
            tokens.append("<sep>")
            word_end_mask.append(1)
            tokens.append("<cls>")
            word_end_mask.append(1)

            input_ids = parser.bert_tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            subword_max_len = max(subword_max_len, len(input_ids))

            all_input_ids[snum, :len(input_ids)] = input_ids
            all_word_end_mask[snum, :len(word_end_mask)] = word_end_mask
            all_input_mask[snum, :len(input_mask)] = input_mask

        all_input_ids = all_input_ids[:, :subword_max_len]
        all_word_end_mask = all_word_end_mask[:, :subword_max_len]
        all_input_mask = all_input_mask[:, :subword_max_len]
        return all_input_ids, all_word_end_mask, all_input_mask

    ########### Evaluate in tf session
    print("evaluating in tf session...")
    subbatch_trees = test_treebank[:2]
    subbatch_sentences = [[leaf.word for leaf in tree.leaves()] for tree in subbatch_trees]
    inp_val_tokens, inp_val_mask, input_mask = bertify_batch([[word for word in sentence] for sentence in subbatch_sentences])
    sess.run(bert_features, {the_inp_tokens: inp_val_tokens, the_inp_mask: input_mask}).shape

    eval_batch_size = 16
    test_predicted = []
    for start_index in tqdm(range(0, len(test_treebank), eval_batch_size)):
        subbatch_trees = test_treebank[start_index:start_index+eval_batch_size]
        subbatch_sentences = [[leaf.word for leaf in tree.leaves()] for tree in subbatch_trees]
        inp_val_tokens, inp_val_mask, _ = bertify_batch([[word for word in sentence] for sentence in subbatch_sentences])
        out_val_chart, out_val_tags = sess.run((the_out_chart, the_out_tags), 
                                        {the_inp_tokens: inp_val_tokens, the_inp_mask: inp_val_mask})
        trees = []
        scores = []
        for snum, sentence in enumerate(subbatch_sentences):
            chart_size = len(sentence) + 1
            tf_chart = out_val_chart[snum,:chart_size,:chart_size,:]
            sentence = list(zip([TAG_VOCAB[idx] for idx in out_val_tags[snum,1:chart_size]], [x for x in sentence]))
            tree, score = parser.decode_from_chart(sentence, tf_chart)
            trees.append(tree)
            scores.append(score)
        test_predicted.extend([p.convert() for p in trees])

    test_fscore = evaluate.evalb('EVALB/', test_treebank[:len(test_predicted)], test_predicted)

    print(str(test_fscore))

    input_node_names = [the_inp_tokens.name.split(':')[0], the_inp_mask.name.split(':')[0]]
    output_node_names = [the_out_chart.name.split(':')[0], the_out_tags.name.split(':')[0]]

    ########### saving and exporting in graph
    print("saving in tf graph...")
    graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names)
    graph_def = TransformGraph(graph_def, input_node_names, output_node_names, [
    'strip_unused_nodes()',
    'remove_nodes(op=Identity, op=CheckNumerics)',
    'fold_constants()',
    'fold_old_batch_norms',
    'fold_batch_norms',
    'round_weights(num_steps=128)',
    ])
    with open(args.export_path + 'xlnet-base.pb', 'wb') as f:
        f.write(graph_def.SerializeToString())

    graph_def = TransformGraph(graph_def, input_node_names, output_node_names,
                            transforms = 
                            ['add_default_attributes',
                'remove_nodes(op=Identity, op=CheckNumerics, op=Dropout)',
                'fold_constants(ignore_errors=true)',
                'fold_batch_norms',
                'fold_old_batch_norms',
                'quantize_weights(fallback_min=-10, fallback_max=10)',
                'strip_unused_nodes',
                'sort_by_execution_order'])
    with open(args.export_path + 'xlnet-base.pb.quantized', 'wb') as f:
        f.write(graph_def.SerializeToString())

    with open(args.export_path + 'vocab-xlnet-base.json', 'w') as fopen:
        json.dump({'label': LABEL_VOCAB, 'tag': TAG_VOCAB}, fopen)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/xlnet_dev=82.50.pt',
                        help='Path to model')
    parser.add_argument('--test_path', type=str, default='data/test-aug.txt',
                    help='Path to test set')
    parser.add_argument('--export_path', type=str, default='export_model/',
                help='Path to export model')

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

# python src/export.py --model_path models/xlnet_dev=82.50.pt --test_path data/test-aug.txt --export_path export_model/