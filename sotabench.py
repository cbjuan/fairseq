from sotabencheval.machine_translation import WMTEvaluator, WMTDataset, Language
from tqdm import tqdm
import torch

class ModelCfg:
    def __init__(self, model_name, arxiv_id, src_lang, dst_lang, hubname, **kwargs):
        self.model_name, self.arxiv_id, self.src_lang, self.dst_lang = model_name, arxiv_id, src_lang, dst_lang
        self.hubname, self.params = hubname, kwargs

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
        model = torch.hub.load('pytorch/fairseq', self.hubname, **self.params)
        model.to("cuda")
        return model


datasets = [
    (WMTDataset.News2014, Language.English, Language.German),
    (WMTDataset.News2014, Language.English, Language.French),
    (WMTDataset.News2019, Language.English, Language.German),
]

models = [
    ModelCfg("ConvS2S", "1705.03122", Language.English, Language.German, 'conv.wmt14.en-de', tokenizer='moses', bpe='subword_nmt'),
    # ModelCfg(Language.English, Language.German, 'transformer.wmt16.en-de', checkpoint_file=?),
    # ModelCfg(Language.English, Language.German, 'conv.wmt17.en-de'),
    # ModelCfg(Language.English, Language.German, 'transformer.wmt18.en-de', checkpoint_file=?),
    ModelCfg("Facebook-FAIR (single)", "1907.06616", Language.English, Language.German, 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe'),
    ModelCfg("Facebook-FAIR (ensemble)", "1907.06616", Language.English, Language.German, 'transformer.wmt19.en-de', tokenizer='moses', bpe='fastbpe', checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt'),

    ModelCfg("ConvS2S", "1705.03122v3", Language.English, Language.French, 'conv.wmt14.en-fr', tokenizer='moses', bpe='subword_nmt'),
    #ModelCfg("Transformer Big", "1806.00187", Language.English, Language.French, 'transformer.wmt14.en-fr', tokenizer='moses', bpe='fastbpe', checkpoint_file=?),
]


for model_cfg in models:
    print("Evaluating model {} ({} -> {})".
          format(model_cfg.model_name, model_cfg.src_lang.name, model_cfg.dst_lang.name))
    model = model_cfg.load_model()
    for ds, src_lang, dst_lang in datasets:
        if src_lang == model_cfg.src_lang and dst_lang == model_cfg.dst_lang:
            evaluator = model_cfg.get_evaluator(ds)

            for index, (sid, text) in enumerate(tqdm(evaluator.metrics.source_segments.items())):
                translated = model.translate(text, beam=50)
                evaluator.add({sid: translated})
                if evaluator.cache_exists:
                    break

            evaluator.save()
            print(evaluator.results)
