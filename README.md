# SentenceBERT - KoBERT

Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks (EMNLP 2019) 논문, kakaobrain 팀이 공개한 KorNLUDatasets 과 ETRI KorBERT를 통해 Korean Sentence BERT를 학습하였습니다.

추가 : [GitHub - BM-K/KoSentenceBERT](https://github.com/BM-K/KoSentenceBERT) 에 있는 내용처럼 KoBERT와 [GitHub - UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers) 를 같이 사용함에 어려움이 있어 최대한 코드 수정을 덜 하는 방향으로 진행했습니다. 

[GitHub - SKTBrain/KoBERT](https://github.com/SKTBrain/KoBERT) 와 [GitHub - UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers) 연동을 위해 [GitHub - monologg/KoBERT-Transformers](https://github.com/monologg/KoBERT-Transformers) 를 사용했습니다.

## Requirements

requirements는 [GitHub - SKTBrain/KoBERT](https://github.com/SKTBrain/KoBERT), [GitHub - UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers), [GitHub - monologg/KoBERT-Transformers](https://github.com/monologg/KoBERT-Transformers)를 참고해 주시기 바랍니다.



## 수정내용

* sentence_transformers_/sentensce_transformers 내 파일 일부 수정(에러 수정, tensorboard 사용 등)

* transformers\tokenization_utils_base.py 1965번째 줄 수정 필요(에러 발생)

  ```python
  vocab_files = self.save_vocabulary(save_directory, filename_prefix=filename_prefix) # 기존
  vocab_files = self.save_vocabulary(save_directory) # 수정
  ```

  * Transformers와 sentence_transformers는 링크를 첨부합니다. 다운로드 후 SentenceEmbedding폴더에 넣어 사용하시면 되겠습니다.
  * [huggingface transformers 수정본](https://drive.google.com/drive/folders/1d-EPl7cIsLisiRzUZ1ijHKmxETWC-oF2?usp=sharing)
  * [sentence_transformers 수정본](https://drive.google.com/drive/folders/1hwK1DOMqGAzMv7vjBjlPMmn8bU7Rhoqv?usp=sharing)

# Train Models

모델 학습을 위해서는 디렉토리에 KorNLUDatasets가 존재해야 합니다. 저는 [GitHub - UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers) 폴더에 있는 KorNLUDatasets 를 사용하였습니다.

pooling mode 는 MEAN-strategy를 사용(default)하였으며, 학습시 모델은`KoBERT_training_NLI.py`, `KoBERT_training_STS.py` 에 있는 `model_save_path`변수를 통해 변경할 수 있습니다.

모든 랜덤 seed는 고정된 상태로 학습을 진행했습니다.

|Model|Cosine Pearson|Cosine Spearman|Euclidean Pearson|Euclidean Spearman|Manhattan Pearson|Manhattan Spearman|Dot Pearson|Dot Spearman|
|:------------------------:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|NLl|63.92|66.82|67.77|66.79|67.70|       66.72        |63.73|64.51|
|STS|**80.39**|**79.30**|76.71|75.72|**76.76**|75.74|75.31|    74.46     |
|STS + NLI|78.88|79.21|**78.14**|**78.28**|78.18|**78.30**|**75.47**|**75.80**|

* 논문과는 다르게 only STS 학습 성능이 NLI+STS보다 좋은 것을 확인할 수 있습니다.

# Example

[GitHub - SKTBrain/KoBERT](https://github.com/SKTBrain/KoBERT) `Application Examples`에 있는 예제를 참고했습니다.

## Semantic Search

SemanticSearch.py는 주어진 문장과 유사한 문장을 찾는 작업입니다. 먼저 Corpus의 모든 문장에 대한 임베딩을 생성합니다.

```python
from sentence_transformers_.sentence_transformers import SentenceTransformer, util
import numpy as np
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--model_path', type=str, default='D:/KoBERT_training/output/STS/KoBERT-2021-04-23_14-43-29', help='model path')

    args = parser.parse_args()


    model_path = args.model_path

    embedder = SentenceTransformer(model_path)

    # Corpus with example sentences
    corpus = ['한 남자가 음식을 먹는다.',
              '한 남자가 빵 한 조각을 먹는다.',
              '그 여자가 아이를 돌본다.',
              '한 남자가 말을 탄다.',
              '한 여자가 바이올린을 연주한다.',
              '두 남자가 수레를 숲 속으로 밀었다.',
              '한 남자가 담으로 싸인 땅에서 백마를 타고 있다.',
              '원숭이 한 마리가 드럼을 연주한다.',
              '치타 한 마리가 먹이 뒤에서 달리고 있다.']

    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

    # Query sentences:
    queries = ['한 남자가 파스타를 먹는다.',
               '고릴라 의상을 입은 누군가가 드럼을 연주하고 있다.',
               '치타가 들판을 가로 질러 먹이를 쫓는다.']

    # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
    top_k = 5
    for query in queries:
        query_embedding = embedder.encode(query, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
        cos_scores = cos_scores.cpu()

        # We use np.argpartition, to only partially sort the top_k results
        top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]

        print("\n\n======================\n\n")
        print("Query:", query)
        print("\nTop 5 most similar sentences in corpus:")

        for idx in top_results[0:top_k]:
            print(corpus[idx].strip(), "(Score: %.4f)" % (cos_scores[idx]))
```

결과는 다음과 같습니다 :

```
======================


Query: 한 남자가 파스타를 먹는다.

Top 5 most similar sentences in corpus:
한 남자가 음식을 먹는다. (Score: 0.5829)
한 남자가 빵 한 조각을 먹는다. (Score: 0.5406)
두 남자가 수레를 숲 속으로 밀었다. (Score: 0.0847)
한 남자가 담으로 싸인 땅에서 백마를 타고 있다. (Score: 0.0649)
그 여자가 아이를 돌본다. (Score: 0.0540)


======================


Query: 고릴라 의상을 입은 누군가가 드럼을 연주하고 있다.

Top 5 most similar sentences in corpus:
원숭이 한 마리가 드럼을 연주한다. (Score: 0.6554)
한 여자가 바이올린을 연주한다. (Score: 0.3377)
치타 한 마리가 먹이 뒤에서 달리고 있다. (Score: 0.2657)
두 남자가 수레를 숲 속으로 밀었다. (Score: 0.2087)
한 남자가 말을 탄다. (Score: 0.1637)


======================


Query: 치타가 들판을 가로 질러 먹이를 쫓는다.

Top 5 most similar sentences in corpus:
치타 한 마리가 먹이 뒤에서 달리고 있다. (Score: 0.8205)
두 남자가 수레를 숲 속으로 밀었다. (Score: 0.3794)
한 남자가 말을 탄다. (Score: 0.2163)
원숭이 한 마리가 드럼을 연주한다. (Score: 0.2029)
한 남자가 담으로 싸인 땅에서 백마를 타고 있다. (Score: 0.2008)
```



