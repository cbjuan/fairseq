import copy
from collections import OrderedDict

from sotabencheval.machine_translation import WMTEvaluator, WMTDataset, Language
from fairseq import utils
from tqdm import tqdm
import torch


class ModelCfg:
    def __init__(self, model_name, arxiv_id, src_lang, dst_lang, hubname, beam, batch_size, **kwargs):
        self.model_name, self.arxiv_id, self.src_lang, self.dst_lang = model_name, arxiv_id, src_lang, dst_lang
        self.hubname, self.beam, self.batch_size, self.params = hubname, beam, batch_size, kwargs

    def get_evaluator(self, dataset):
        return WMTEvaluator(
            dataset=dataset,
            source_lang=self.src_lang,
            target_lang=self.dst_lang,
            local_root="data/nlp/wmt",
            model_name=self.model_name,
            paper_arxiv_id=self.arxiv_id
        )

    def load_model(self):
        model = torch.hub.load('pytorch/fairseq', self.hubname, beam=self.beam, **self.params)
        model.to("cuda")
        return model


def translate_batch(model, sids, sentences, beam=5):
    input = [model.encode(sentence) for sentence in sentences]
    lengths = [len(t) for t in input]
    dataset = model.task.build_dataset_for_inference(input, lengths)
    samples = dataset.collater(dataset)
    samples = utils.apply_to_sample(
        lambda tensor: tensor.to(model.device),
        samples
    )
    ids = samples['id'].cpu()

    gen_args = copy.copy(model.args)
    gen_args.beam = beam
    generator = model.task.build_generator(gen_args)

    translations = model.task.inference_step(generator, model.models, samples)
    hypos = [translation[0]['tokens'] for translation in translations]
    translated = [model.decode(hypo) for hypo in hypos]
    return OrderedDict([(sids[id], tr) for id, tr in zip(ids, translated)])


def batchify(items, batch_size):
    items = list(items)
    items = sorted(items, key=lambda x: len(x[1]), reverse=True)
    length = len(items)
    return [items[i * batch_size: (i+1) * batch_size] for i in range((length + batch_size - 1) // batch_size)]


datasets = [
    (WMTDataset.News2014, Language.English, Language.German),
    (WMTDataset.News2014, Language.English, Language.French),
    (WMTDataset.News2019, Language.English, Language.German),
]

models = [
    # English -> German models
    ModelCfg("ConvS2S", "1705.03122", Language.English, Language.German, 'conv.wmt14.en-de', 5, 128,
             tokenizer='moses', bpe='subword_nmt'),

    # ModelCfg(Language.English, Language.German, 'transformer.wmt16.en-de', checkpoint_file=?),
    # ModelCfg(Language.English, Language.German, 'conv.wmt17.en-de'),
    # ModelCfg(Language.English, Language.German, 'transformer.wmt18.en-de', checkpoint_file=?),

    ModelCfg("Facebook-FAIR (single)", "1907.06616", Language.English, Language.German,
             'transformer.wmt19.en-de.single_model', 50, 64, tokenizer='moses', bpe='fastbpe'),

    ModelCfg("Facebook-FAIR (ensemble)", "1907.06616", Language.English, Language.German,
             'transformer.wmt19.en-de', 50, 64, tokenizer='moses', bpe='fastbpe',
             checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt'),

    # English -> French models
    ModelCfg("ConvS2S", "1705.03122v3", Language.English, Language.French,
             'conv.wmt14.en-fr', 5, 128, tokenizer='moses', bpe='subword_nmt'),
#    ModelCfg("Transformer Big", "1806.00187", Language.English, Language.French,
#             'transformer.wmt14.en-fr', 50, 64, tokenizer='moses', bpe='fastbpe'),
]


for model_cfg in models:
    print("Evaluating model {} ({} -> {})".
          format(model_cfg.model_name, model_cfg.src_lang.name, model_cfg.dst_lang.name))
    model = model_cfg.load_model()
    for ds, src_lang, dst_lang in datasets:
        if src_lang == model_cfg.src_lang and dst_lang == model_cfg.dst_lang:
            evaluator = model_cfg.get_evaluator(ds)

            iter = tqdm(batchify(evaluator.metrics.source_segments.items(), model_cfg.batch_size))
            for batch in iter:
                sids, texts = zip(*batch)
                answers = translate_batch(model, sids, texts, beam=model_cfg.beam)
                evaluator.add(answers)
                if evaluator.cache_exists:
                    break

            evaluator.save()
            print(evaluator.results)
