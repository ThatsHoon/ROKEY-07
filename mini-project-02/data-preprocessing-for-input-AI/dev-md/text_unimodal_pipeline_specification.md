# 텍스트 단일 모달리티 학습 파이프라인 명세서

> **버전**: 1.0  
> **목적**: AI 학습용 텍스트 임베딩 데이터셋 생성  
> **핵심 방식**: Transformer 기반 텍스트 인코딩  

---

## 1. 프로젝트 개요

### 1.1 목표

텍스트 데이터를 **고차원 벡터(Latent Space)**로 인코딩하여 AI 모델 학습에 활용할 수 있는 임베딩 데이터셋을 생성한다.

### 1.2 핵심 설계 원칙

| 원칙 | 설명 |
|-----|------|
| **확장성** | 다양한 텍스트 인코더로 교체 가능 |
| **모듈성** | 전처리, 인코딩, 출력 컴포넌트 독립적 |
| **효율성** | 배치 처리 및 분산 학습 지원 |
| **품질** | 텍스트 정제 및 품질 필터링 |

### 1.3 지원 텍스트 유형

| 유형 | 설명 | 예시 |
|-----|------|------|
| 문장 | 단일 문장 | 리뷰, 트윗, 캡션 |
| 문단 | 여러 문장 | 뉴스 기사, 블로그 |
| 문서 | 긴 텍스트 | 논문, 보고서, 책 |
| 대화 | 멀티턴 대화 | 챗봇 로그, 인터뷰 |
| QA 쌍 | 질문-답변 | SQuAD, 고객 문의 |

---

## 2. 전체 아키텍처

### 2.1 시스템 다이어그램

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              RAW TEXT DATA                                      │
│                                                                                  │
│         문장, 문단, 문서, 대화, QA 쌍 등 다양한 텍스트 형식                         │
│                                                                                  │
└──────────────────────────────────┬──────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 1: 텍스트 전처리 (Preprocessing)                    │
│                                                                                  │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│   │   정제       │  │  정규화      │  │  토크나이징   │  │  패딩/트렁케이션│        │
│   │  Cleaning    │  │ Normalization│  │ Tokenization │  │ Padding/Trunc │        │
│   └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘        │
│                                                                                  │
│   - HTML 태그 제거                - 유니코드 정규화      - BPE/WordPiece    - 최대 길이 조정   │
│   - 특수문자 처리                 - 소문자 변환 (옵션)   - 서브워드 분리    - [PAD] 토큰 추가  │
│   - 중복 공백 제거                - 숫자 정규화          - Special 토큰     - Attention Mask   │
│                                                                                  │
└──────────────────────────────────┬──────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 2: 텍스트 인코딩 (Text Encoding)                    │
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                     Pre-trained Text Encoder                            │   │
│   │                                                                         │   │
│   │    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐            │   │
│   │    │  BERT   │ or │ RoBERTa │ or │   GPT   │ or │ Custom  │            │   │
│   │    └─────────┘    └─────────┘    └─────────┘    └─────────┘            │   │
│   │                                                                         │   │
│   │    Input: Token IDs [B, Seq]                                           │   │
│   │    Output: Hidden States [B, Seq, Hidden_Dim]                          │   │
│   │                                                                         │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└──────────────────────────────────┬──────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 3: 표현 추출 (Representation Extraction)            │
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                      Pooling Strategy                                   │   │
│   │                                                                         │   │
│   │    ┌───────────────┐  ┌───────────────┐  ┌───────────────┐             │   │
│   │    │  [CLS] Token  │  │  Mean Pooling │  │ Attention Pool│             │   │
│   │    │    Pooling    │  │               │  │               │             │   │
│   │    └───────────────┘  └───────────────┘  └───────────────┘             │   │
│   │                                                                         │   │
│   │    [B, Seq, D] → [B, D]                                                │   │
│   │                                                                         │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└──────────────────────────────────┬──────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 4: 출력 투영 (Output Projection)                    │
│                                                                                  │
│                    ┌───────────────────────────────────┐                        │
│                    │   Linear + LayerNorm + Activation │                        │
│                    │                                   │                        │
│                    │   [B, Hidden_Dim] → [B, Output_Dim]│                        │
│                    └───────────────────────────────────┘                        │
│                                                                                  │
│                              Final Text Embedding                               │
│                                                                                  │
└──────────────────────────────────┬──────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 5: 저장 및 출력                                     │
│                                                                                  │
│         .npy / .h5 / .parquet / .safetensors → AI Training Dataset             │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 데이터 흐름 요약

```
Raw Text → Cleaning → Tokenizing → Encoding → Pooling → Projection → Output
    │          │           │           │          │          │          │
    │          │           │           │          │          │          │
  원본       정제       토큰화      특징추출    벡터화     차원조정     저장
```

---

## 3. 단계별 상세 명세

### 3.1 Phase 1: 텍스트 전처리

#### 3.1.1 텍스트 정제 (Cleaning)

| 작업 | 설명 | 예시 |
|-----|------|------|
| HTML 제거 | 태그 및 엔티티 제거 | `<p>Hello</p>` → `Hello` |
| URL 처리 | 제거 또는 토큰화 | `https://...` → `[URL]` |
| 이메일 처리 | 제거 또는 마스킹 | `user@mail.com` → `[EMAIL]` |
| 특수문자 | 불필요한 문자 제거 | `Hello!!!` → `Hello!` |
| 공백 정규화 | 연속 공백 단일화 | `Hello   World` → `Hello World` |
| 이모지 처리 | 제거 또는 텍스트 변환 | `😀` → `:smile:` 또는 제거 |

```python
import re
import html
from typing import Optional

class TextCleaner:
    """텍스트 정제 유틸리티"""
    
    # 정규표현식 패턴
    HTML_PATTERN = re.compile(r'<[^>]+>')
    URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
    EMAIL_PATTERN = re.compile(r'\S+@\S+\.\S+')
    WHITESPACE_PATTERN = re.compile(r'\s+')
    REPEATED_PUNCT_PATTERN = re.compile(r'([!?.]){2,}')
    
    def __init__(
        self,
        remove_html: bool = True,
        remove_urls: bool = False,
        url_token: Optional[str] = "[URL]",
        remove_emails: bool = False,
        email_token: Optional[str] = "[EMAIL]",
        normalize_whitespace: bool = True,
        normalize_punctuation: bool = True,
        lowercase: bool = False
    ):
        self.remove_html = remove_html
        self.remove_urls = remove_urls
        self.url_token = url_token
        self.remove_emails = remove_emails
        self.email_token = email_token
        self.normalize_whitespace = normalize_whitespace
        self.normalize_punctuation = normalize_punctuation
        self.lowercase = lowercase
    
    def clean(self, text: str) -> str:
        """텍스트 정제 수행"""
        if not text:
            return ""
        
        # HTML 엔티티 디코딩
        text = html.unescape(text)
        
        # HTML 태그 제거
        if self.remove_html:
            text = self.HTML_PATTERN.sub('', text)
        
        # URL 처리
        if self.remove_urls:
            text = self.URL_PATTERN.sub('', text)
        elif self.url_token:
            text = self.URL_PATTERN.sub(self.url_token, text)
        
        # 이메일 처리
        if self.remove_emails:
            text = self.EMAIL_PATTERN.sub('', text)
        elif self.email_token:
            text = self.EMAIL_PATTERN.sub(self.email_token, text)
        
        # 반복 구두점 정규화
        if self.normalize_punctuation:
            text = self.REPEATED_PUNCT_PATTERN.sub(r'\1', text)
        
        # 공백 정규화
        if self.normalize_whitespace:
            text = self.WHITESPACE_PATTERN.sub(' ', text).strip()
        
        # 소문자 변환
        if self.lowercase:
            text = text.lower()
        
        return text
```

#### 3.1.2 토크나이징 (Tokenization)

| 토크나이저 | 알고리즘 | 특징 | 대표 모델 |
|-----------|---------|------|----------|
| **WordPiece** | 서브워드 분리 | OOV 처리 우수 | BERT |
| **BPE** | Byte Pair Encoding | 빈도 기반 병합 | GPT, RoBERTa |
| **SentencePiece** | 언어 독립적 | 다국어 지원 | XLNet, T5 |
| **Unigram** | 확률 기반 | 최적 분할 탐색 | ALBERT |

```python
from transformers import AutoTokenizer
from typing import Dict, List, Union
import torch

class TextTokenizer:
    """텍스트 토크나이저 래퍼"""
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = 512,
        padding: str = "max_length",
        truncation: bool = True,
        return_tensors: str = "pt"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.return_tensors = return_tensors
    
    def tokenize(
        self, 
        texts: Union[str, List[str]]
    ) -> Dict[str, torch.Tensor]:
        """
        텍스트를 토큰화
        
        Args:
            texts: 단일 텍스트 또는 텍스트 리스트
            
        Returns:
            {
                'input_ids': [B, Seq],
                'attention_mask': [B, Seq],
                'token_type_ids': [B, Seq]  # BERT 계열
            }
        """
        if isinstance(texts, str):
            texts = [texts]
        
        encoded = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors=self.return_tensors
        )
        
        return encoded
    
    def decode(self, token_ids: torch.Tensor) -> List[str]:
        """토큰 ID를 텍스트로 디코딩"""
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
    
    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)
    
    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id
    
    @property
    def cls_token_id(self) -> int:
        return self.tokenizer.cls_token_id
    
    @property
    def sep_token_id(self) -> int:
        return self.tokenizer.sep_token_id
```

#### 3.1.3 전처리 파이프라인 통합

```python
class TextPreprocessor:
    """텍스트 전처리 통합 파이프라인"""
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = 512,
        cleaning_config: dict = None
    ):
        self.cleaner = TextCleaner(**(cleaning_config or {}))
        self.tokenizer = TextTokenizer(
            model_name=model_name,
            max_length=max_length
        )
    
    def preprocess(
        self, 
        texts: Union[str, List[str]]
    ) -> Dict[str, torch.Tensor]:
        """
        전체 전처리 수행: 정제 → 토크나이징
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # 정제
        cleaned_texts = [self.cleaner.clean(text) for text in texts]
        
        # 토크나이징
        encoded = self.tokenizer.tokenize(cleaned_texts)
        
        return encoded
    
    def __call__(self, texts):
        return self.preprocess(texts)
```

---

### 3.2 Phase 2: 텍스트 인코딩

#### 3.2.1 인코더 추상 인터페이스

```python
from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class TextEncoder(ABC, nn.Module):
    """
    텍스트 인코더 추상 인터페이스
    다양한 사전학습 모델을 통일된 인터페이스로 사용
    """
    
    @abstractmethod
    def encode(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: 토큰 ID [batch_size, seq_len]
            attention_mask: 어텐션 마스크 [batch_size, seq_len]
            
        Returns:
            hidden_states: [batch_size, seq_len, hidden_dim]
        """
        pass
    
    @property
    @abstractmethod
    def hidden_dim(self) -> int:
        """인코더 출력 차원"""
        pass
    
    @property
    @abstractmethod
    def max_length(self) -> int:
        """최대 시퀀스 길이"""
        pass
```

#### 3.2.2 사전학습 인코더 비교

| 인코더 | 출력 차원 | 최대 길이 | 특징 | 용도 |
|-------|----------|----------|------|------|
| **BERT-base** | 768 | 512 | 양방향 인코딩 | 범용 NLU |
| **BERT-large** | 1024 | 512 | 더 큰 용량 | 고성능 NLU |
| **RoBERTa** | 768/1024 | 512 | 개선된 사전학습 | 강건한 표현 |
| **ALBERT** | 128~4096 | 512 | 파라미터 효율적 | 경량화 |
| **DistilBERT** | 768 | 512 | BERT 증류 | 빠른 추론 |
| **ELECTRA** | 256/768 | 512 | 판별자 사전학습 | 효율적 학습 |
| **Sentence-BERT** | 384/768 | 512 | 문장 임베딩 특화 | 유사도 계산 |
| **Longformer** | 768 | 4096 | 긴 문서 처리 | 문서 인코딩 |
| **BigBird** | 768 | 4096 | 희소 어텐션 | 긴 시퀀스 |

#### 3.2.3 인코더 구현

```python
from transformers import AutoModel, AutoConfig
import torch.nn as nn

class BERTEncoder(TextEncoder):
    """BERT 기반 텍스트 인코더"""
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        pretrained: bool = True,
        freeze: bool = False
    ):
        super().__init__()
        
        if pretrained:
            self.model = AutoModel.from_pretrained(model_name)
        else:
            config = AutoConfig.from_pretrained(model_name)
            self.model = AutoModel.from_config(config)
        
        self._hidden_dim = self.model.config.hidden_size
        self._max_length = self.model.config.max_position_embeddings
        
        if freeze:
            self._freeze_parameters()
    
    def _freeze_parameters(self):
        """인코더 파라미터 동결"""
        for param in self.model.parameters():
            param.requires_grad = False
    
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        텍스트 인코딩 수행
        
        Returns:
            [batch_size, seq_len, hidden_dim]
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.last_hidden_state
    
    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim
    
    @property
    def max_length(self) -> int:
        return self._max_length


class SentenceTransformerEncoder(TextEncoder):
    """Sentence-BERT 기반 인코더 (문장 임베딩 특화)"""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        super().__init__()
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self._hidden_dim = self.model.get_sentence_embedding_dimension()
        self._max_length = self.model.max_seq_length
    
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        # SentenceTransformer는 자체 풀링 수행
        # 시퀀스 출력이 필요한 경우 내부 모델 직접 사용
        features = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        return self.model(features)['sentence_embedding']
    
    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim
    
    @property
    def max_length(self) -> int:
        return self._max_length


class LongformerEncoder(TextEncoder):
    """Longformer 기반 인코더 (긴 문서용)"""
    
    def __init__(
        self,
        model_name: str = "allenai/longformer-base-4096",
        pretrained: bool = True
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name) if pretrained else None
        self._hidden_dim = self.model.config.hidden_size
        self._max_length = self.model.config.max_position_embeddings
    
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        # Longformer는 global attention mask 필요
        global_attention_mask = torch.zeros_like(input_ids)
        global_attention_mask[:, 0] = 1  # [CLS] 토큰에 global attention
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask
        )
        return outputs.last_hidden_state
    
    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim
    
    @property
    def max_length(self) -> int:
        return self._max_length
```

---

### 3.3 Phase 3: 표현 추출 (Pooling)

#### 3.3.1 Pooling 전략 비교

| 전략 | 설명 | 장점 | 단점 |
|-----|------|-----|------|
| **[CLS] Token** | 첫 번째 토큰 사용 | 간단, BERT 표준 | 정보 집중도 의존 |
| **Mean Pooling** | 모든 토큰 평균 | 균형적 표현 | 패딩 처리 필요 |
| **Max Pooling** | 토큰별 최대값 | 중요 특징 강조 | 정보 손실 가능 |
| **Attention Pooling** | 학습된 가중 평균 | 적응적 | 추가 파라미터 |
| **Last Token** | 마지막 토큰 | GPT 스타일 | 위치 의존적 |

#### 3.3.2 Pooling 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextPooler(nn.Module):
    """텍스트 시퀀스 풀링 모듈"""
    
    def __init__(
        self,
        hidden_dim: int,
        pooling_strategy: str = "cls",  # cls, mean, max, attention
        dropout: float = 0.1
    ):
        super().__init__()
        self.pooling_strategy = pooling_strategy
        self.hidden_dim = hidden_dim
        
        # Attention Pooling용 레이어
        if pooling_strategy == "attention":
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, Seq, D]
            attention_mask: [B, Seq]
            
        Returns:
            pooled: [B, D]
        """
        if self.pooling_strategy == "cls":
            # [CLS] 토큰 (첫 번째 위치)
            pooled = hidden_states[:, 0]
            
        elif self.pooling_strategy == "mean":
            # Mean Pooling (마스크 고려)
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                hidden_states = hidden_states * mask
                pooled = hidden_states.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                pooled = hidden_states.mean(dim=1)
                
        elif self.pooling_strategy == "max":
            # Max Pooling
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                hidden_states = hidden_states.masked_fill(mask == 0, -1e9)
            pooled = hidden_states.max(dim=1)[0]
            
        elif self.pooling_strategy == "attention":
            # Attention Pooling
            attn_scores = self.attention(hidden_states)  # [B, Seq, 1]
            if attention_mask is not None:
                attn_scores = attn_scores.masked_fill(
                    attention_mask.unsqueeze(-1) == 0, -1e9
                )
            attn_weights = F.softmax(attn_scores, dim=1)
            pooled = (hidden_states * attn_weights).sum(dim=1)
            
        elif self.pooling_strategy == "last":
            # Last Token (GPT 스타일)
            if attention_mask is not None:
                # 마지막 유효 토큰 위치 찾기
                seq_lengths = attention_mask.sum(dim=1) - 1
                batch_size = hidden_states.size(0)
                pooled = hidden_states[
                    torch.arange(batch_size, device=hidden_states.device),
                    seq_lengths
                ]
            else:
                pooled = hidden_states[:, -1]
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
        return self.dropout(pooled)
```

---

### 3.4 Phase 4: 출력 투영 (Projection)

```python
class OutputProjection(nn.Module):
    """최종 임베딩 차원 투영"""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: str = "tanh",  # tanh, gelu, relu, none
        normalize: bool = True
    ):
        super().__init__()
        
        self.projection = nn.Linear(input_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim) if normalize else nn.Identity()
        
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, input_dim]
        Returns:
            [B, output_dim]
        """
        x = self.projection(x)
        x = self.layer_norm(x)
        x = self.activation(x)
        return x
```

---

### 3.5 Phase 5: 저장 및 출력

#### 3.5.1 출력 스키마

```python
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np

@dataclass
class TextEmbeddingDataPoint:
    """텍스트 임베딩 데이터 포인트"""
    sample_id: str                      # 고유 식별자
    text_vector: np.ndarray             # 텍스트 임베딩 [output_dim]
    original_text: Optional[str]        # 원본 텍스트 (선택적)
    metadata: Dict[str, Any]            # 추가 메타데이터
    
    def to_dict(self) -> dict:
        return {
            'sample_id': self.sample_id,
            'text_vector': self.text_vector.tolist(),
            'original_text': self.original_text,
            'metadata': self.metadata
        }
```

#### 3.5.2 저장 유틸리티

```python
import h5py
import pandas as pd
import numpy as np
from typing import List

class TextDatasetExporter:
    """텍스트 임베딩 데이터셋 내보내기"""
    
    @staticmethod
    def to_hdf5(data_points: List[TextEmbeddingDataPoint], path: str):
        """HDF5 형식으로 저장"""
        with h5py.File(path, 'w') as f:
            # 벡터 저장
            vectors = np.stack([dp.text_vector for dp in data_points])
            f.create_dataset('text_vectors', data=vectors, compression='gzip')
            
            # ID 저장
            ids = [dp.sample_id for dp in data_points]
            f.create_dataset('sample_ids', data=np.array(ids, dtype='S'))
            
            # 원본 텍스트 저장 (선택적)
            if data_points[0].original_text:
                texts = [dp.original_text for dp in data_points]
                f.create_dataset('original_texts', data=np.array(texts, dtype='S'))
    
    @staticmethod
    def to_parquet(data_points: List[TextEmbeddingDataPoint], path: str):
        """Parquet 형식으로 저장"""
        records = [dp.to_dict() for dp in data_points]
        df = pd.DataFrame(records)
        df.to_parquet(path, compression='snappy')
    
    @staticmethod
    def to_numpy(data_points: List[TextEmbeddingDataPoint], path: str):
        """NumPy 형식으로 저장"""
        vectors = np.stack([dp.text_vector for dp in data_points])
        np.save(path, vectors)
```

---

## 4. 전체 파이프라인 통합

### 4.1 End-to-End Pipeline 클래스

```python
class TextEmbeddingPipeline(nn.Module):
    """
    텍스트 임베딩 생성 파이프라인
    Raw Text → Text Embedding Vector
    """
    
    def __init__(
        self,
        encoder_name: str = "bert-base-uncased",
        max_length: int = 512,
        output_dim: int = 768,
        pooling_strategy: str = "mean",
        freeze_encoder: bool = False
    ):
        super().__init__()
        
        # 1. 전처리기
        self.preprocessor = TextPreprocessor(
            model_name=encoder_name,
            max_length=max_length
        )
        
        # 2. 인코더
        self.encoder = BERTEncoder(
            model_name=encoder_name,
            freeze=freeze_encoder
        )
        
        # 3. 풀러
        self.pooler = TextPooler(
            hidden_dim=self.encoder.hidden_dim,
            pooling_strategy=pooling_strategy
        )
        
        # 4. 출력 투영
        self.output_projection = OutputProjection(
            input_dim=self.encoder.hidden_dim,
            output_dim=output_dim
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        토큰화된 입력으로부터 임베딩 생성
        
        Args:
            input_ids: [B, Seq]
            attention_mask: [B, Seq]
            
        Returns:
            text_embedding: [B, output_dim]
        """
        # 인코딩
        hidden_states = self.encoder.encode(input_ids, attention_mask)
        
        # 풀링
        pooled = self.pooler(hidden_states, attention_mask)
        
        # 출력 투영
        embedding = self.output_projection(pooled)
        
        return embedding
    
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        원본 텍스트로부터 임베딩 생성 (편의 메서드)
        
        Args:
            texts: 텍스트 리스트
            
        Returns:
            text_embeddings: [B, output_dim]
        """
        # 전처리
        encoded = self.preprocessor(texts)
        
        # 디바이스 이동
        device = next(self.parameters()).device
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # 임베딩 생성
        return self.forward(input_ids, attention_mask)
```

### 4.2 학습 데이터 생성 스크립트

```python
from torch.utils.data import DataLoader
from tqdm import tqdm

def generate_text_embeddings(
    pipeline: TextEmbeddingPipeline,
    data_loader: DataLoader,
    output_path: str,
    device: torch.device = torch.device('cuda'),
    save_original_text: bool = False
):
    """
    전체 데이터셋에 대해 텍스트 임베딩 생성 및 저장
    """
    pipeline.eval()
    pipeline.to(device)
    
    all_vectors = []
    all_metadata = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Generating text embeddings"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # 임베딩 생성
            embeddings = pipeline(input_ids, attention_mask)
            
            all_vectors.append(embeddings.cpu().numpy())
            all_metadata.extend(batch['metadata'])
    
    # 저장
    vectors = np.concatenate(all_vectors, axis=0)
    
    data_points = [
        TextEmbeddingDataPoint(
            sample_id=meta['id'],
            text_vector=vec,
            original_text=meta.get('text') if save_original_text else None,
            metadata=meta
        )
        for vec, meta in zip(vectors, all_metadata)
    ]
    
    TextDatasetExporter.to_hdf5(data_points, output_path)
    print(f"Saved {len(data_points)} text embeddings to {output_path}")
```

---

## 5. 품질 관리

### 5.1 텍스트 품질 평가

```python
from typing import Tuple
import re

class TextQualityAssessor:
    """텍스트 품질 평가"""
    
    def __init__(
        self,
        min_length: int = 10,
        max_length: int = 10000,
        min_word_count: int = 3,
        max_repeat_ratio: float = 0.3
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.min_word_count = min_word_count
        self.max_repeat_ratio = max_repeat_ratio
    
    def assess(self, text: str) -> Tuple[float, dict]:
        """
        텍스트 품질 점수 및 상세 정보 반환
        
        Returns:
            (quality_score, details_dict)
        """
        details = {}
        score = 1.0
        
        # 길이 체크
        length = len(text)
        details['length'] = length
        if length < self.min_length:
            score *= 0.5
            details['length_issue'] = 'too_short'
        elif length > self.max_length:
            score *= 0.8
            details['length_issue'] = 'too_long'
        
        # 단어 수 체크
        words = text.split()
        word_count = len(words)
        details['word_count'] = word_count
        if word_count < self.min_word_count:
            score *= 0.5
            details['word_count_issue'] = 'too_few'
        
        # 반복 비율 체크
        if words:
            unique_words = set(w.lower() for w in words)
            repeat_ratio = 1 - (len(unique_words) / len(words))
            details['repeat_ratio'] = repeat_ratio
            if repeat_ratio > self.max_repeat_ratio:
                score *= (1 - repeat_ratio)
                details['repeat_issue'] = True
        
        # 특수문자 비율
        special_chars = len(re.findall(r'[^a-zA-Z0-9가-힣\s]', text))
        special_ratio = special_chars / max(length, 1)
        details['special_char_ratio'] = special_ratio
        if special_ratio > 0.3:
            score *= 0.7
        
        # 언어 감지 (간단한 휴리스틱)
        korean_chars = len(re.findall(r'[가-힣]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        if korean_chars > english_chars:
            details['detected_language'] = 'ko'
        else:
            details['detected_language'] = 'en'
        
        details['quality_score'] = score
        return score, details
    
    def filter(self, text: str, threshold: float = 0.5) -> bool:
        """품질 기준 통과 여부"""
        score, _ = self.assess(text)
        return score >= threshold
```

### 5.2 필터링 적용

```python
QUALITY_THRESHOLDS = {
    'min_score': 0.5,
    'min_length': 10,
    'max_length': 10000,
    'min_words': 3
}

def filter_texts(
    texts: List[str],
    assessor: TextQualityAssessor,
    threshold: float = 0.5
) -> List[Tuple[int, str]]:
    """
    품질 기준을 통과한 텍스트만 반환
    
    Returns:
        [(original_index, text), ...]
    """
    filtered = []
    for i, text in enumerate(texts):
        if assessor.filter(text, threshold):
            filtered.append((i, text))
    return filtered
```

---

## 6. 학습 데이터셋

### 6.1 데이터셋 플랫폼

| 플랫폼 | URL | 특징 |
|-------|-----|------|
| **Hugging Face Datasets** | https://huggingface.co/datasets | 가장 활발, Python API |
| **Kaggle** | https://kaggle.com/datasets | 경진대회 데이터 |
| **Papers With Code** | https://paperswithcode.com/datasets | 논문 연계 |

### 6.2 주요 텍스트 데이터셋

| 데이터셋 | 규모 | 내용 | 라이선스 | 링크 |
|---------|-----|------|---------|------|
| **SQuAD 2.0** | 150,000 QA | 독해 기반 질의응답 | CC-BY-SA 4.0 | https://rajpurkar.github.io/SQuAD-explorer |
| **The Pile** | 800GB | LLM 사전학습용 | MIT | https://pile.eleuther.ai |
| **Common Crawl** | 페타바이트급 | 웹 크롤링 | 다양 | https://commoncrawl.org |
| **Wikipedia Dumps** | 다국어 | 위키피디아 전체 | CC-BY-SA | https://dumps.wikimedia.org |
| **RedPajama** | 1.2조 토큰 | LLaMA 재현용 | Apache 2.0 | https://github.com/togethercomputer/RedPajama-Data |
| **OpenWebText2** | 69GB | Reddit 링크 기반 | 연구용 | https://openwebtext2.readthedocs.io |
| **C4** | 750GB | 정제된 Common Crawl | ODC-BY | https://huggingface.co/datasets/c4 |
| **BookCorpus** | 11,000권 | 소설 텍스트 | 연구용 | 요청 필요 |
| **CC-News** | 7600만 기사 | 뉴스 아카이브 | 연구용 | https://commoncrawl.org/blog/news-dataset-available |

### 6.3 SQuAD 데이터셋 상세

```
구성:
├── train-v2.0.json     # 130,319 QA 쌍
└── dev-v2.0.json       # 11,873 QA 쌍

JSON 구조:
{
  "data": [{
    "title": "Article Title",
    "paragraphs": [{
      "context": "문단 텍스트...",
      "qas": [{
        "question": "질문?",
        "id": "unique_id",
        "answers": [{"text": "답변", "answer_start": 42}],
        "is_impossible": false
      }]
    }]
  }]
}

특징:
- SQuAD 2.0은 답변 불가능한 질문 포함
- 위키피디아 문서 기반
- 다운로드: ~35MB
```

**활용 코드:**
```python
from datasets import load_dataset

# SQuAD 2.0 로드
squad = load_dataset("squad_v2", split="train")

# 샘플 구조
sample = squad[0]
context = sample["context"]      # 문맥 문단
question = sample["question"]    # 질문
answers = sample["answers"]      # {"text": [...], "answer_start": [...]}

# 텍스트 추출 예시
texts = [item["context"] for item in squad]
```

### 6.4 한국어 텍스트 데이터셋

| 데이터셋 | 규모 | 내용 | 링크 |
|---------|-----|------|------|
| **AI Hub 텍스트** | 대규모 | 다양한 도메인 | https://aihub.or.kr |
| **모두의 말뭉치** | 다양 | 국립국어원 코퍼스 | https://corpus.korean.go.kr |
| **KorQuAD 2.0** | 100,000+ QA | 한국어 SQuAD | https://korquad.github.io |
| **KLUE** | 8종 벤치마크 | NLU 태스크 | https://klue-benchmark.com |
| **Naver NER** | 90,000 문장 | 개체명 인식 | https://github.com/naver/nlp-challenge |
| **NSMC** | 200,000 리뷰 | 감성 분류 | https://github.com/e9t/nsmc |

**AI Hub 주요 텍스트 데이터:**
```
- 한국어 대화 데이터 (100만+ 대화)
- 기계독해 데이터 (6만+ QA)
- 법률/의료 전문 말뭉치
- 뉴스 기사 코퍼스
- 번역 병렬 코퍼스 (한-영, 한-중, 한-일)
```

### 6.5 데이터 로딩 유틸리티

```python
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    """텍스트 데이터셋 래퍼"""
    
    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        text_column: str = "text",
        max_samples: int = None
    ):
        self.dataset = load_dataset(dataset_name, split=split)
        if max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
        self.text_column = text_column
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            'text': item[self.text_column],
            'metadata': {'id': str(idx), **item}
        }


def create_text_dataloader(
    dataset_name: str,
    preprocessor: TextPreprocessor,
    batch_size: int = 32,
    split: str = "train"
) -> DataLoader:
    """텍스트 데이터로더 생성"""
    
    dataset = TextDataset(dataset_name, split=split)
    
    def collate_fn(batch):
        texts = [item['text'] for item in batch]
        metadata = [item['metadata'] for item in batch]
        
        encoded = preprocessor(texts)
        encoded['metadata'] = metadata
        
        return encoded
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
```

### 6.6 데이터셋 선택 가이드

| 용도 | 권장 데이터셋 | 이유 |
|-----|-------------|------|
| **프로토타입** | SQuAD, NSMC | 소규모, 잘 정제됨 |
| **문장 임베딩** | STS Benchmark, NLI | 유사도 학습용 |
| **문서 임베딩** | Wikipedia, C4 | 긴 텍스트 |
| **대화 모델** | AI Hub 대화, DailyDialog | 멀티턴 구조 |
| **한국어 특화** | KorQuAD, KLUE | 한국어 품질 보장 |
| **대규모 사전학습** | The Pile, RedPajama | 최대 규모 |

---

## 7. 프로젝트 구조

```
text-embedding-pipeline/
├── configs/
│   ├── base.yaml                 # 기본 설정
│   ├── encoder/
│   │   ├── bert.yaml
│   │   ├── roberta.yaml
│   │   └── longformer.yaml
│   └── training.yaml
│
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── cleaner.py            # TextCleaner
│   │   └── tokenizer.py          # TextTokenizer
│   │
│   ├── encoder/
│   │   ├── __init__.py
│   │   ├── base.py               # TextEncoder ABC
│   │   ├── bert.py               # BERTEncoder
│   │   ├── sentence_transformer.py
│   │   └── longformer.py
│   │
│   ├── pooling/
│   │   ├── __init__.py
│   │   └── pooler.py             # TextPooler
│   │
│   ├── projection/
│   │   ├── __init__.py
│   │   └── output.py             # OutputProjection
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py            # TextDataset
│   │   └── exporter.py           # TextDatasetExporter
│   │
│   ├── pipeline.py               # TextEmbeddingPipeline
│   ├── quality.py                # TextQualityAssessor
│   └── utils.py
│
├── scripts/
│   ├── preprocess.py
│   ├── generate_embeddings.py
│   └── evaluate.py
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_encoder.py
│   └── test_pipeline.py
│
├── notebooks/
│   └── exploration.ipynb
│
├── requirements.txt
└── README.md
```

---

## 8. 기술 스택

| 영역 | 권장 도구 | 버전 |
|-----|----------|-----|
| 프레임워크 | PyTorch | 2.0+ |
| Transformer | HuggingFace Transformers | 4.30+ |
| 토크나이저 | HuggingFace Tokenizers | 최신 |
| 문장 임베딩 | Sentence-Transformers | 최신 |
| 데이터 로딩 | HuggingFace Datasets | 최신 |
| 저장 | h5py, pyarrow | 최신 |
| 실험 관리 | Weights & Biases | 최신 |

---

## 9. 구현 로드맵

### Phase 1: MVP (1주)
- [x] 텍스트 전처리 파이프라인
- [x] BERT 인코더 통합
- [x] Mean Pooling 구현
- [x] HDF5 출력

### Phase 2: 확장 (1주)
- [ ] 다양한 인코더 지원 (RoBERTa, Longformer)
- [ ] 다양한 Pooling 전략
- [ ] 품질 필터링

### Phase 3: 최적화 (1주)
- [ ] Mixed Precision 적용
- [ ] 배치 처리 최적화
- [ ] 분산 처리 지원

### Phase 4: 검증 (3일)
- [ ] 임베딩 품질 평가 (유사도, 클러스터링)
- [ ] Downstream task 테스트
- [ ] 문서화 완료

---

## 10. 참고 자료

### 관련 연구
- BERT (Devlin et al., 2019) - 양방향 사전학습
- Sentence-BERT (Reimers et al., 2019) - 문장 임베딩
- SimCSE (Gao et al., 2021) - 대조 학습 기반 임베딩
- E5 (Wang et al., 2022) - 범용 텍스트 임베딩

### 유용한 링크
- HuggingFace Transformers: https://huggingface.co/docs/transformers
- Sentence-Transformers: https://www.sbert.net
- HuggingFace Datasets: https://huggingface.co/docs/datasets

---

> **문서 작성**: Claude  
> **최종 수정**: 2025년 12월