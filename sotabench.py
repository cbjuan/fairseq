from sotabencheval.machine_translation import WMTEvaluator, TranslationMetrics, Language
from tqdm import tqdm
import torch


evaluator = WMTEvaluator(
    source_lang=Language.English,
    target_lang=Language.German,
    local_root="data/nlp/wmt",
    model_name="Facebook-FAIR (single)",
    paper_arxiv_id="1907.06616"
)

model = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
model.to("cuda")

for index, (sid, text) in tqdm(enumerate(evaluator.metrics.source_sentences.items())):
    translated = model.translate(text)
    evaluator.add({sid: translated})
    if index == 31 and evaluator.cache_exists:
        break

evaluator.save()
print(evaluator.results)
