# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3

import json

import datasets

_CITATION = '''
@misc{bge-m3,
      title={BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation}, 
      author={Jianlv Chen and Shitao Xiao and Peitian Zhang and Kun Luo and Defu Lian and Zheng Liu},
      year={2024},
      eprint={2402.03216},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
'''

_LANGUAGES = [
    'ar',
    'de',
    'en',
    'es',
    'fr',
    'hi',
    'it',
    'ja',
    'ko',
    'pt',
    'ru',
    'th',
    'zh',
]

_DESCRIPTION = 'dataset load script for MLDR'

_DATASET_URLS = {
    lang: {
        'train': f'https://hf-mirror.com/datasets/Shitao/MLDR/resolve/main/mldr-v1.0-{lang}/train.jsonl.gz',
        'dev': f'https://hf-mirror.com/datasets/Shitao/MLDR/resolve/main/mldr-v1.0-{lang}/dev.jsonl.gz',
        'test': f'https://hf-mirror.com/datasets/Shitao/MLDR/resolve/main/mldr-v1.0-{lang}/test.jsonl.gz',
    } for lang in _LANGUAGES
}

_DATASET_CORPUS_URLS = {
    f'corpus-{lang}': {
        'corpus': f'https://hf-mirror.com/datasets/Shitao/MLDR/resolve/main/mldr-v1.0-{lang}/corpus.jsonl.gz'
    } for lang in _LANGUAGES
}


class MLDR(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [datasets.BuilderConfig(
            version=datasets.Version('1.0.0'),
            name=lang, description=f'MLDR dataset in language {lang}.'
        ) for lang in _LANGUAGES
    ] + [
        datasets.BuilderConfig(
            version=datasets.Version('1.0.0'),
            name=f'corpus-{lang}', description=f'corpus of MLDR dataset in language {lang}.'
        ) for lang in _LANGUAGES
    ]

    def _info(self):
        name = self.config.name
        if name.startswith('corpus-'):
            features = datasets.Features({
                'docid': datasets.Value('string'),
                'text': datasets.Value('string'),
            })
        else:
            features = datasets.Features({
                'query_id': datasets.Value('string'),
                'query': datasets.Value('string'),
                'positive_passages': [{
                    'docid': datasets.Value('string'),
                    'text': datasets.Value('string'),
                }],
                'negative_passages': [{
                    'docid': datasets.Value('string'),
                    'text': datasets.Value('string'),
                }],
            })

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage='https://github.com/FlagOpen/FlagEmbedding',
            # License for the dataset if available
            license='mit',
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        name = self.config.name
        if name.startswith('corpus-'):
            downloaded_files = dl_manager.download_and_extract(_DATASET_CORPUS_URLS[name])
            splits = [
                datasets.SplitGenerator(
                    name='corpus',
                    gen_kwargs={
                        'filepath': downloaded_files['corpus'],
                    },
                ),
            ]
        else:
            downloaded_files = dl_manager.download_and_extract(_DATASET_URLS[name])
            splits = [
                datasets.SplitGenerator(
                    name='train',
                    gen_kwargs={
                        'filepath': downloaded_files['train'],
                    },
                ),
                datasets.SplitGenerator(
                    name='dev',
                    gen_kwargs={
                        'filepath': downloaded_files['dev'],
                    },
                ),
                datasets.SplitGenerator(
                    name='test',
                    gen_kwargs={
                        'filepath': downloaded_files['test'],
                    },
                ),
            ]
        return splits

    def _generate_examples(self, filepath):
        name = self.config.name
        if name.startswith('corpus-'):
            with open(filepath, encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    yield data['docid'], data
        else:
            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    qid = data['query_id']
                    for feature in ['negative_passages', 'positive_passages']:
                        if data.get(feature) is None:
                            data[feature] = []
                    yield qid, data
