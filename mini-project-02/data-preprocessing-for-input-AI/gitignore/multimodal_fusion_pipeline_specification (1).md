# 확장 가능한 멀티모달 데이터 통합 파이프라인 명세서

> **버전**: 1.0  
> **목적**: AI 학습용 통합 임베딩 데이터셋 생성  
> **핵심 방식**: Cross-Attention Transformer Fusion  

---

## 1. 프로젝트 개요

### 1.1 목표

다양한 모달리티(시각, 청각, 텍스트, 촉각 등)의 데이터를 **공통 임베딩 공간(Shared Latent Space)**에서 하나의 고차원 벡터로 통합하여 AI 모델 학습에 활용할 수 있는 데이터셋을 생성한다.

### 1.2 핵심 설계 원칙

| 원칙 | 설명 |
|-----|------|
| **확장성** | 새로운 모달리티를 최소한의 코드 변경으로 추가 가능 |
| **모듈성** | 각 컴포넌트(인코더, 융합, 출력)가 독립적으로 교체 가능 |
| **유연성** | 일부 모달리티가 없는 샘플도 처리 가능 |
| **효율성** | 메모리 최적화 및 분산 처리 지원 |

### 1.3 지원 모달리티

| 모달리티 | 데이터 유형 | 예시 |
|---------|-----------|------|
| Vision | 이미지, 비디오 프레임 | RGB 이미지, 의료 영상 |
| Audio | 음성, 사운드 | 발화, 환경음, 음악 |
| Text | 자연어 | 문장, 문서, 캡션 |
| Tactile | 촉각 센서 데이터 | 압력, 진동, 온도 |
| *확장* | 사용자 정의 | EEG, IMU, LiDAR 등 |

---

## 2. 전체 아키텍처

### 2.1 시스템 다이어그램

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              RAW MULTIMODAL DATA                                │
├────────────┬────────────┬────────────┬────────────┬─────────────────────────────┤
│   Vision   │   Audio    │    Text    │  Tactile   │    ... (확장 모달리티)       │
└─────┬──────┴─────┬──────┴─────┬──────┴─────┬──────┴─────────────────────────────┘
      │            │            │            │
      ▼            ▼            ▼            ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 1: 모달리티별 전처리                                │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                        │
│   │ Resize   │  │ Resample │  │ Tokenize │  │ Calibrate│                        │
│   │ Normalize│  │ MFCC/Mel │  │ Clean    │  │ Segment  │                        │
│   │ Augment  │  │ Pad/Trim │  │ Pad      │  │ Normalize│                        │
│   └──────────┘  └──────────┘  └──────────┘  └──────────┘                        │
└─────────────────────────────────────────────────────────────────────────────────┘
      │            │            │            │
      ▼            ▼            ▼            ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 2: 개별 인코딩 (Unimodal Encoding)                  │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                        │
│   │ ViT /    │  │ Wav2Vec /│  │ BERT /   │  │ Custom   │                        │
│   │ CLIP     │  │ Whisper  │  │ RoBERTa  │  │ Encoder  │                        │
│   │ DINOv2   │  │ HuBERT   │  │ S-BERT   │  │ CNN-LSTM │                        │
│   └──────────┘  └──────────┘  └──────────┘  └──────────┘                        │
│        ↓              ↓             ↓             ↓                             │
│   [B,S₁,D₁]     [B,S₂,D₂]     [B,S₃,D₃]    [B,S₄,D₄]                           │
└─────────────────────────────────────────────────────────────────────────────────┘
      │            │            │            │
      ▼            ▼            ▼            ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 3: 차원 정렬 (Projection)                           │
│                                                                                  │
│              Linear Projection → Shared Dimension (D_shared)                    │
│                                                                                  │
│   [B,S₁,D₁] → [B,S₁,D]    [B,S₂,D₂] → [B,S₂,D]    ...                          │
└─────────────────────────────────────────────────────────────────────────────────┘
      │            │            │            │
      └────────────┴─────┬──────┴────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 4: Cross-Attention Fusion                          │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                Multi-Head Cross-Attention Transformer                     │  │
│  │                                                                           │  │
│  │    ┌─────────┐         ┌─────────────────┐         ┌─────────┐           │  │
│  │    │Modality │──Query──│  Cross-Attention│──Value──│Modality │           │  │
│  │    │    A    │         │     Layer       │         │    B    │           │  │
│  │    └─────────┘         └─────────────────┘         └─────────┘           │  │
│  │                                                                           │  │
│  │              ← Bidirectional / Pairwise Attention →                       │  │
│  │                                                                           │  │
│  │    ┌──────────────────────────────────────────────────────────┐          │  │
│  │    │  Feed-Forward Network + Layer Normalization + Residual  │          │  │
│  │    └──────────────────────────────────────────────────────────┘          │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 5: 통합 표현 생성                                   │
│                                                                                  │
│       ┌─────────────────────────────────────────────────────────────┐           │
│       │  Pooling Strategy: [CLS] Token / Mean Pooling / Attention  │           │
│       └─────────────────────────────────────────────────────────────┘           │
│                                    │                                            │
│                                    ▼                                            │
│                    ┌───────────────────────────────┐                            │
│                    │  Unified Vector [B, D_output] │                            │
│                    │     (Shared Latent Space)     │                            │
│                    └───────────────────────────────┘                            │
└─────────────────────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 6: 저장 및 출력                                     │
│                                                                                  │
│         .npy / .h5 / .parquet / .safetensors → AI Training Dataset             │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 데이터 흐름 요약

```
Raw Data → Preprocessing → Encoding → Projection → Fusion → Pooling → Output
   │            │              │           │          │          │        │
   │            │              │           │          │          │        │
  원본        정규화         특징추출     차원통일    교차주의   벡터화   저장
```

---

## 3. 단계별 상세 명세

### 3.1 Phase 1: 모달리티별 전처리

#### 3.1.1 Vision (시각)

| 항목 | 설정 |
|-----|------|
| 입력 형식 | JPEG, PNG, BMP, WEBP |
| 해상도 정규화 | 224×224 (ViT 기준) 또는 384×384 |
| 정규화 | ImageNet mean/std 또는 데이터셋 통계 |
| 증강 (학습 시) | RandomCrop, HorizontalFlip, ColorJitter |
| 출력 형태 | `Tensor [B, 3, H, W]` |

```python
# Vision 전처리 파이프라인
vision_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

#### 3.1.2 Audio (청각)

| 항목 | 설정 |
|-----|------|
| 입력 형식 | WAV, MP3, FLAC |
| 샘플링 레이트 | 16,000 Hz (표준) |
| 특징 추출 | Mel-Spectrogram 또는 Raw Waveform |
| 시퀀스 길이 | 최대 30초 (패딩/트렁케이션) |
| 출력 형태 | `Tensor [B, T, n_mels]` 또는 `[B, samples]` |

```python
# Audio 전처리 파이프라인
def preprocess_audio(waveform, sr=16000, max_length=480000):
    # 리샘플링
    if original_sr != sr:
        waveform = torchaudio.transforms.Resample(original_sr, sr)(waveform)
    
    # 길이 정규화
    if waveform.shape[1] > max_length:
        waveform = waveform[:, :max_length]
    else:
        waveform = F.pad(waveform, (0, max_length - waveform.shape[1]))
    
    return waveform
```

#### 3.1.3 Text (텍스트)

| 항목 | 설정 |
|-----|------|
| 입력 형식 | 문자열 (UTF-8) |
| 토크나이저 | BPE, WordPiece, SentencePiece |
| 최대 시퀀스 | 512 tokens (BERT 기준) |
| 특수 토큰 | [CLS], [SEP], [PAD] |
| 출력 형태 | `Tensor [B, seq_len]` (token IDs) |

```python
# Text 전처리 파이프라인
def preprocess_text(text, tokenizer, max_length=512):
    encoded = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    return encoded['input_ids'], encoded['attention_mask']
```

#### 3.1.4 Tactile (촉각)

| 항목 | 설정 |
|-----|------|
| 입력 형식 | 센서 시계열 데이터 (CSV, NPY) |
| 센서 수 | 가변 (예: 6축 IMU, 압력 어레이) |
| 샘플링 레이트 | 센서 의존적 (100Hz~1000Hz) |
| 정규화 | Min-Max 또는 Z-score |
| 출력 형태 | `Tensor [B, T, n_sensors]` |

```python
# Tactile 전처리 파이프라인
def preprocess_tactile(sensor_data, target_length=1000):
    # 캘리브레이션 오프셋 적용
    calibrated = sensor_data - calibration_offset
    
    # 정규화
    normalized = (calibrated - mean) / std
    
    # 시퀀스 길이 조정
    if len(normalized) > target_length:
        normalized = normalized[:target_length]
    else:
        normalized = np.pad(normalized, ((0, target_length - len(normalized)), (0, 0)))
    
    return torch.tensor(normalized, dtype=torch.float32)
```

---

### 3.2 Phase 2: 개별 모달리티 인코딩

#### 3.2.1 인코더 추상 인터페이스

```python
from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class ModalityEncoder(ABC, nn.Module):
    """
    모든 모달리티 인코더가 구현해야 하는 추상 인터페이스
    확장 시 이 클래스를 상속하여 새로운 모달리티 지원
    """
    
    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 전처리된 입력 데이터
        Returns:
            Tensor [batch_size, sequence_length, hidden_dim]
        """
        pass
    
    @property
    @abstractmethod
    def output_dim(self) -> int:
        """인코더 출력 차원"""
        pass
    
    @property
    @abstractmethod
    def modality_name(self) -> str:
        """모달리티 식별자"""
        pass
```

#### 3.2.2 권장 사전학습 인코더

| 모달리티 | 인코더 | 출력 차원 | 특징 |
|---------|-------|----------|------|
| Vision | ViT-B/16 | 768 | 패치 기반, 위치 인코딩 포함 |
| Vision | CLIP ViT-L/14 | 1024 | 멀티모달 사전학습 |
| Vision | DINOv2 | 768/1024 | 자기지도 학습, 강건함 |
| Audio | Wav2Vec 2.0 | 768 | 자기지도 음성 표현 |
| Audio | Whisper Encoder | 512/1024 | 다국어, 강건함 |
| Audio | HuBERT | 768 | 마스크 예측 기반 |
| Text | BERT-base | 768 | 범용 언어 이해 |
| Text | RoBERTa-large | 1024 | 개선된 BERT |
| Text | Sentence-BERT | 384/768 | 문장 임베딩 특화 |
| Tactile | Custom CNN-LSTM | 256/512 | 도메인 특화 설계 필요 |

#### 3.2.3 인코더 구현 예시

```python
class VisionEncoder(ModalityEncoder):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self._output_dim = self.model.embed_dim
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # ViT: [B, num_patches + 1, embed_dim]
        features = self.model.forward_features(x)
        return features
    
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    @property
    def modality_name(self) -> str:
        return "vision"


class AudioEncoder(ModalityEncoder):
    def __init__(self, model_name='facebook/wav2vec2-base-960h'):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self._output_dim = self.model.config.hidden_size
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(x)
        return outputs.last_hidden_state
    
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    @property
    def modality_name(self) -> str:
        return "audio"


class TextEncoder(ModalityEncoder):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self._output_dim = self.model.config.hidden_size
        
    def encode(self, x: torch.Tensor, attention_mask=None) -> torch.Tensor:
        outputs = self.model(input_ids=x, attention_mask=attention_mask)
        return outputs.last_hidden_state
    
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    @property
    def modality_name(self) -> str:
        return "text"
```

---

### 3.3 Phase 3: 차원 정렬 (Projection Layer)

각 모달리티 인코더의 출력 차원이 다르므로, 융합 전에 공통 차원으로 투영합니다.

```python
class ProjectionLayer(nn.Module):
    """
    모달리티별 인코더 출력을 공통 차원으로 투영
    """
    def __init__(
        self, 
        input_dim: int, 
        shared_dim: int = 768,
        dropout: float = 0.1
    ):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim]
        Returns:
            [batch, seq_len, shared_dim]
        """
        return self.projection(x)


class ModalityProjector(nn.Module):
    """
    모든 모달리티에 대한 투영 레이어 관리
    """
    def __init__(self, encoder_dims: dict, shared_dim: int = 768):
        super().__init__()
        self.projectors = nn.ModuleDict({
            name: ProjectionLayer(dim, shared_dim)
            for name, dim in encoder_dims.items()
        })
    
    def forward(self, modality_name: str, x: torch.Tensor) -> torch.Tensor:
        return self.projectors[modality_name](x)
```

---

### 3.4 Phase 4: Cross-Attention Fusion

#### 3.4.1 Cross-Attention 메커니즘

```python
class CrossModalAttention(nn.Module):
    """
    두 모달리티 간 교차 주의 집중 레이어
    Query: 한 모달리티의 표현
    Key/Value: 다른 모달리티의 표현
    """
    def __init__(
        self, 
        dim: int, 
        num_heads: int = 8, 
        dropout: float = 0.1,
        qkv_bias: bool = True
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self, 
        query: torch.Tensor, 
        key_value: torch.Tensor,
        query_mask: torch.Tensor = None,
        kv_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            query: [B, Sq, D] - Query 모달리티
            key_value: [B, Skv, D] - Key/Value 모달리티
            query_mask: [B, Sq] - Query 패딩 마스크
            kv_mask: [B, Skv] - Key/Value 패딩 마스크
        Returns:
            [B, Sq, D] - Cross-attended 표현
        """
        B, Sq, D = query.shape
        _, Skv, _ = key_value.shape
        
        # Multi-head projection
        Q = self.q_proj(query).view(B, Sq, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key_value).view(B, Skv, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(key_value).view(B, Skv, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        
        # Masking (optional)
        if kv_mask is not None:
            attn = attn.masked_fill(
                kv_mask.unsqueeze(1).unsqueeze(2) == 0, 
                float('-inf')
            )
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        out = (attn @ V).transpose(1, 2).reshape(B, Sq, D)
        out = self.out_proj(out)
        
        # Residual + LayerNorm
        query = self.norm1(query + out)
        
        # Feed-forward
        query = self.norm2(query + self.ffn(query))
        
        return query
```

#### 3.4.2 융합 전략

| 전략 | 설명 | 복잡도 | 적합한 경우 |
|-----|------|-------|-----------|
| **Pairwise Bidirectional** | 모든 모달리티 쌍에 양방향 attention | O(n²) | 모달리티 수가 적을 때 |
| **Anchor-based** | 주 모달리티를 기준으로 융합 | O(n) | 주 모달리티가 명확할 때 |
| **Sequential** | 순차적으로 모달리티 추가 | O(n) | 계층적 정보 구조 |
| **Perceiver-style** | Latent Query로 모든 모달리티 압축 | O(n) | 모달리티 수가 많을 때 |

#### 3.4.3 융합 모듈 구현

```python
class MultimodalFusionTransformer(nn.Module):
    """
    여러 모달리티를 Cross-Attention으로 융합하는 Transformer
    """
    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        fusion_strategy: str = 'pairwise'  # 'pairwise', 'anchor', 'perceiver'
    ):
        super().__init__()
        self.dim = dim
        self.fusion_strategy = fusion_strategy
        
        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            CrossModalAttention(dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Self-attention for final fusion
        self.self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            for _ in range(num_layers // 2)
        ])
        
        # Learnable modality tokens (optional)
        self.modality_tokens = nn.ParameterDict()
    
    def register_modality(self, name: str):
        """새 모달리티 토큰 등록"""
        self.modality_tokens[name] = nn.Parameter(torch.randn(1, 1, self.dim))
    
    def forward(
        self, 
        modality_features: dict,  # {name: [B, S, D]}
        modality_masks: dict = None  # {name: [B, S]}
    ) -> torch.Tensor:
        """
        Args:
            modality_features: 모달리티별 투영된 특징
            modality_masks: 모달리티별 패딩 마스크
        Returns:
            [B, S_total, D] - 융합된 표현
        """
        modality_list = list(modality_features.keys())
        features = list(modality_features.values())
        masks = modality_masks or {k: None for k in modality_list}
        
        if self.fusion_strategy == 'pairwise':
            # Pairwise bidirectional cross-attention
            for layer in self.cross_attn_layers:
                updated_features = []
                for i, (name_i, feat_i) in enumerate(modality_features.items()):
                    for j, (name_j, feat_j) in enumerate(modality_features.items()):
                        if i != j:
                            feat_i = layer(feat_i, feat_j, masks[name_i], masks[name_j])
                    updated_features.append(feat_i)
                modality_features = dict(zip(modality_list, updated_features))
        
        # Concatenate all modality features
        fused = torch.cat(list(modality_features.values()), dim=1)
        
        # Apply self-attention
        for layer in self.self_attn_layers:
            fused = layer(fused)
        
        return fused
```

---

### 3.5 Phase 5: 통합 표현 생성

#### 3.5.1 Pooling 전략

```python
class UnifiedRepresentationHead(nn.Module):
    """
    융합된 시퀀스를 단일 벡터로 변환
    """
    def __init__(
        self, 
        input_dim: int = 768, 
        output_dim: int = 768,
        pooling_strategy: str = 'cls'  # 'cls', 'mean', 'attention'
    ):
        super().__init__()
        self.pooling_strategy = pooling_strategy
        
        # [CLS] 토큰 (cls 전략용)
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))
        
        # Attention pooling (attention 전략용)
        self.attention_pool = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # 최종 투영
        self.output_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.Tanh()
        )
    
    def forward(self, fused_features: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            fused_features: [B, S, D] - 융합된 시퀀스
            mask: [B, S] - 패딩 마스크
        Returns:
            [B, output_dim] - 통합 벡터
        """
        if self.pooling_strategy == 'cls':
            # [CLS] 토큰을 시퀀스 앞에 추가
            B = fused_features.size(0)
            cls_tokens = self.cls_token.expand(B, -1, -1)
            fused_features = torch.cat([cls_tokens, fused_features], dim=1)
            pooled = fused_features[:, 0]  # [CLS] 위치
            
        elif self.pooling_strategy == 'mean':
            if mask is not None:
                mask = mask.unsqueeze(-1)
                fused_features = fused_features * mask
                pooled = fused_features.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                pooled = fused_features.mean(dim=1)
                
        elif self.pooling_strategy == 'attention':
            attn_weights = self.attention_pool(fused_features)  # [B, S, 1]
            if mask is not None:
                attn_weights = attn_weights.masked_fill(~mask.unsqueeze(-1), float('-inf'))
                attn_weights = F.softmax(attn_weights, dim=1)
            pooled = (fused_features * attn_weights).sum(dim=1)
        
        return self.output_proj(pooled)
```

---

### 3.6 Phase 6: 출력 및 저장

#### 3.6.1 출력 스키마

```python
@dataclass
class UnifiedDataPoint:
    """통합 데이터 포인트 스키마"""
    sample_id: str                          # 고유 식별자
    unified_vector: np.ndarray              # 통합 임베딩 [output_dim]
    modality_mask: Dict[str, bool]          # 포함된 모달리티
    modality_vectors: Dict[str, np.ndarray] # 개별 모달리티 벡터 (선택적)
    metadata: Dict[str, Any]                # 추가 메타데이터
    
    def to_dict(self) -> dict:
        return {
            'sample_id': self.sample_id,
            'unified_vector': self.unified_vector.tolist(),
            'modality_mask': self.modality_mask,
            'metadata': self.metadata
        }
```

#### 3.6.2 저장 형식 비교

| 형식 | 장점 | 단점 | 권장 용도 |
|-----|-----|-----|---------|
| **NPY/NPZ** | 간단, NumPy 호환 | 메타데이터 제한 | 프로토타이핑 |
| **HDF5** | 대용량, 계층적 | 병렬 쓰기 제한 | 연구/실험 |
| **Parquet** | 컬럼 기반, 압축 | 복잡한 중첩 구조 어려움 | 프로덕션 |
| **Safetensors** | 빠른 로딩, 안전 | 메타데이터 제한 | 모델 배포 |

#### 3.6.3 저장 유틸리티

```python
class DatasetExporter:
    """통합 데이터셋 내보내기"""
    
    @staticmethod
    def to_hdf5(data_points: List[UnifiedDataPoint], path: str):
        with h5py.File(path, 'w') as f:
            # 벡터 저장
            vectors = np.stack([dp.unified_vector for dp in data_points])
            f.create_dataset('unified_vectors', data=vectors, compression='gzip')
            
            # 메타데이터 저장
            ids = [dp.sample_id for dp in data_points]
            f.create_dataset('sample_ids', data=np.array(ids, dtype='S'))
            
            # 모달리티 마스크 저장
            modalities = list(data_points[0].modality_mask.keys())
            for mod in modalities:
                mask = [dp.modality_mask[mod] for dp in data_points]
                f.create_dataset(f'mask_{mod}', data=np.array(mask))
    
    @staticmethod
    def to_parquet(data_points: List[UnifiedDataPoint], path: str):
        records = [dp.to_dict() for dp in data_points]
        df = pd.DataFrame(records)
        df.to_parquet(path, compression='snappy')
```

---

## 4. 확장 가이드

### 4.1 새 모달리티 추가 절차

```
1. ModalityEncoder 서브클래스 구현
   └─ encode(), output_dim, modality_name 정의

2. 전처리 함수 정의
   └─ 원본 데이터 → 인코더 입력 형태 변환

3. Registry에 등록
   └─ ModalityRegistry.register(name, encoder, preprocessor)

4. ProjectionLayer 자동 생성
   └─ ModalityProjector가 새 차원 처리

5. FusionModule이 자동 인식
   └─ 추가 코드 변경 불필요
```

### 4.2 Registry 패턴 구현

```python
class ModalityRegistry:
    """모달리티 인코더 및 전처리기 중앙 관리"""
    
    _encoders: Dict[str, Type[ModalityEncoder]] = {}
    _preprocessors: Dict[str, Callable] = {}
    _configs: Dict[str, dict] = {}
    
    @classmethod
    def register(
        cls,
        name: str,
        encoder_class: Type[ModalityEncoder],
        preprocessor: Callable,
        config: dict = None
    ):
        cls._encoders[name] = encoder_class
        cls._preprocessors[name] = preprocessor
        cls._configs[name] = config or {}
    
    @classmethod
    def get_encoder(cls, name: str, **kwargs) -> ModalityEncoder:
        config = {**cls._configs[name], **kwargs}
        return cls._encoders[name](**config)
    
    @classmethod
    def get_preprocessor(cls, name: str) -> Callable:
        return cls._preprocessors[name]
    
    @classmethod
    def available_modalities(cls) -> List[str]:
        return list(cls._encoders.keys())


# 사용 예시: 새 모달리티 등록
ModalityRegistry.register(
    name='lidar',
    encoder_class=LiDAREncoder,
    preprocessor=preprocess_lidar,
    config={'num_points': 16384}
)
```

### 4.3 확장 예시: EEG 모달리티

```python
class EEGEncoder(ModalityEncoder):
    """뇌파 신호 인코더"""
    
    def __init__(self, num_channels=64, hidden_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(num_channels, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True),
            num_layers=4
        )
        self._output_dim = hidden_dim
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, channels, time_steps]
        x = self.conv(x)  # [B, 256, T]
        x = x.transpose(1, 2)  # [B, T, 256]
        x = self.transformer(x)
        return x
    
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    @property
    def modality_name(self) -> str:
        return "eeg"


# 등록
ModalityRegistry.register(
    name='eeg',
    encoder_class=EEGEncoder,
    preprocessor=preprocess_eeg,
    config={'num_channels': 64}
)
```

---

## 5. 전체 파이프라인 통합

### 5.1 End-to-End Pipeline 클래스

```python
class MultimodalFusionPipeline(nn.Module):
    """
    전체 멀티모달 융합 파이프라인
    Raw Data → Unified Vector
    """
    
    def __init__(
        self,
        modalities: List[str],
        shared_dim: int = 768,
        output_dim: int = 768,
        num_fusion_layers: int = 4,
        fusion_strategy: str = 'pairwise',
        pooling_strategy: str = 'cls'
    ):
        super().__init__()
        
        self.modalities = modalities
        self.shared_dim = shared_dim
        
        # 1. 모달리티별 인코더
        self.encoders = nn.ModuleDict({
            mod: ModalityRegistry.get_encoder(mod)
            for mod in modalities
        })
        
        # 2. 투영 레이어
        encoder_dims = {mod: enc.output_dim for mod, enc in self.encoders.items()}
        self.projector = ModalityProjector(encoder_dims, shared_dim)
        
        # 3. 융합 모듈
        self.fusion = MultimodalFusionTransformer(
            dim=shared_dim,
            num_layers=num_fusion_layers,
            fusion_strategy=fusion_strategy
        )
        
        # 4. 통합 표현 헤드
        self.unified_head = UnifiedRepresentationHead(
            input_dim=shared_dim,
            output_dim=output_dim,
            pooling_strategy=pooling_strategy
        )
    
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            inputs: {modality_name: preprocessed_tensor}
            masks: {modality_name: padding_mask}
        Returns:
            unified_vector: [B, output_dim]
        """
        masks = masks or {}
        
        # Phase 2: 인코딩
        encoded = {}
        for mod, x in inputs.items():
            encoded[mod] = self.encoders[mod].encode(x)
        
        # Phase 3: 투영
        projected = {}
        for mod, feat in encoded.items():
            projected[mod] = self.projector(mod, feat)
        
        # Phase 4: 융합
        fused = self.fusion(projected, masks)
        
        # Phase 5: 통합 표현
        unified = self.unified_head(fused)
        
        return unified
    
    def process_batch(
        self,
        raw_inputs: Dict[str, Any]
    ) -> torch.Tensor:
        """원본 데이터부터 처리하는 편의 메서드"""
        preprocessed = {}
        for mod, raw in raw_inputs.items():
            preprocess_fn = ModalityRegistry.get_preprocessor(mod)
            preprocessed[mod] = preprocess_fn(raw)
        
        return self.forward(preprocessed)
```

### 5.2 학습 데이터 생성 스크립트

```python
def generate_training_dataset(
    pipeline: MultimodalFusionPipeline,
    data_loader: DataLoader,
    output_path: str,
    device: torch.device = torch.device('cuda')
):
    """
    전체 데이터셋에 대해 통합 벡터 생성 및 저장
    """
    pipeline.eval()
    pipeline.to(device)
    
    all_vectors = []
    all_metadata = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Generating unified vectors"):
            inputs = {k: v.to(device) for k, v in batch['inputs'].items()}
            masks = {k: v.to(device) for k, v in batch.get('masks', {}).items()}
            
            # 통합 벡터 생성
            unified = pipeline(inputs, masks)
            
            all_vectors.append(unified.cpu().numpy())
            all_metadata.extend(batch['metadata'])
    
    # 저장
    vectors = np.concatenate(all_vectors, axis=0)
    
    data_points = [
        UnifiedDataPoint(
            sample_id=meta['id'],
            unified_vector=vec,
            modality_mask=meta['modality_mask'],
            metadata=meta
        )
        for vec, meta in zip(vectors, all_metadata)
    ]
    
    DatasetExporter.to_hdf5(data_points, output_path)
    print(f"Saved {len(data_points)} samples to {output_path}")
```

---

## 6. Missing Modality 처리

### 6.1 전략 개요

일부 샘플에서 특정 모달리티가 누락된 경우의 처리 방법:

| 전략 | 설명 | 장단점 |
|-----|------|-------|
| **Zero Masking** | 없는 모달리티는 0으로 처리 | 간단하지만 정보 손실 |
| **Learnable Placeholder** | 학습 가능한 기본 토큰 사용 | 유연하지만 학습 필요 |
| **Modality Dropout** | 학습 시 랜덤하게 모달리티 제거 | 강건성 향상 |
| **Conditional Generation** | 다른 모달리티로부터 예측 | 복잡하지만 효과적 |

### 6.2 구현

```python
class MissingModalityHandler(nn.Module):
    """누락된 모달리티 처리"""
    
    def __init__(self, modalities: List[str], dim: int):
        super().__init__()
        # 모달리티별 학습 가능한 placeholder
        self.placeholders = nn.ParameterDict({
            mod: nn.Parameter(torch.randn(1, 1, dim) * 0.02)
            for mod in modalities
        })
    
    def forward(
        self,
        features: Dict[str, torch.Tensor],
        available: Dict[str, bool],
        seq_lengths: Dict[str, int]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: 인코딩된 특징
            available: 모달리티 가용성
            seq_lengths: 대체 시퀀스 길이
        Returns:
            완성된 특징 딕셔너리
        """
        result = {}
        batch_size = next(iter(features.values())).size(0)
        
        for mod in self.placeholders.keys():
            if available.get(mod, False) and mod in features:
                result[mod] = features[mod]
            else:
                # Placeholder로 대체
                placeholder = self.placeholders[mod].expand(
                    batch_size, 
                    seq_lengths.get(mod, 1), 
                    -1
                )
                result[mod] = placeholder
        
        return result
```

---

## 7. 최적화 전략

### 7.1 메모리 최적화

| 기법 | 적용 방법 |
|-----|---------|
| **Gradient Checkpointing** | 중간 활성화 재계산으로 메모리 절약 |
| **Mixed Precision (AMP)** | FP16/BF16 연산으로 메모리 50% 절감 |
| **Flash Attention** | 효율적인 attention 구현 |
| **Gradient Accumulation** | 작은 배치로 큰 effective batch size |

```python
# AMP 적용 예시
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    unified = pipeline(inputs)
    loss = criterion(unified, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 7.2 분산 학습

```python
# DDP 설정
model = DistributedDataParallel(
    pipeline,
    device_ids=[local_rank],
    find_unused_parameters=True
)

# DeepSpeed ZeRO 적용
deepspeed_config = {
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu"}
    },
    "fp16": {"enabled": True}
}
```

---

## 8. 품질 관리

### 8.1 모달리티별 품질 점수

```python
def compute_quality_scores(
    sample: Dict[str, Any]
) -> Dict[str, float]:
    """각 모달리티의 품질 점수 계산"""
    scores = {}
    
    # Vision: 블러, 노이즈 검사
    if 'vision' in sample:
        scores['vision'] = assess_image_quality(sample['vision'])
    
    # Audio: SNR 계산
    if 'audio' in sample:
        scores['audio'] = compute_snr(sample['audio'])
    
    # Text: 길이, 문법 점수
    if 'text' in sample:
        scores['text'] = assess_text_quality(sample['text'])
    
    return scores
```

### 8.2 필터링 기준

```python
QUALITY_THRESHOLDS = {
    'vision': 0.6,   # 이미지 품질 점수
    'audio': 10.0,   # SNR (dB)
    'text': 0.7,     # 문법/품질 점수
    'overall': 0.5   # 전체 평균
}

def filter_low_quality(sample, scores, thresholds=QUALITY_THRESHOLDS):
    """품질 기준 미달 샘플 필터링"""
    for mod, score in scores.items():
        if mod in thresholds and score < thresholds[mod]:
            return False
    return True
```

---

## 9. 프로젝트 구조

```
multimodal-fusion-pipeline/
├── configs/
│   ├── base.yaml                 # 기본 설정
│   ├── encoders/
│   │   ├── vision.yaml
│   │   ├── audio.yaml
│   │   ├── text.yaml
│   │   └── tactile.yaml
│   ├── fusion.yaml               # 융합 설정
│   └── training.yaml             # 학습 설정
│
├── src/
│   ├── __init__.py
│   ├── encoders/
│   │   ├── __init__.py
│   │   ├── base.py               # ModalityEncoder ABC
│   │   ├── vision.py
│   │   ├── audio.py
│   │   ├── text.py
│   │   ├── tactile.py
│   │   └── custom/               # 사용자 정의 인코더
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── vision_transforms.py
│   │   ├── audio_transforms.py
│   │   ├── text_transforms.py
│   │   └── tactile_transforms.py
│   │
│   ├── fusion/
│   │   ├── __init__.py
│   │   ├── cross_attention.py    # CrossModalAttention
│   │   ├── fusion_transformer.py # MultimodalFusionTransformer
│   │   ├── projection.py         # ProjectionLayer
│   │   └── pooling.py            # UnifiedRepresentationHead
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py            # MultimodalDataset
│   │   ├── collator.py           # 배치 콜레이터
│   │   └── exporter.py           # DatasetExporter
│   │
│   ├── pipeline.py               # MultimodalFusionPipeline
│   ├── registry.py               # ModalityRegistry
│   └── utils/
│       ├── quality.py            # 품질 평가
│       └── optimization.py       # 최적화 유틸리티
│
├── scripts/
│   ├── preprocess.py             # 전처리 실행
│   ├── encode.py                 # 인코딩 실행
│   ├── generate_dataset.py       # 데이터셋 생성
│   └── evaluate.py               # 평가
│
├── tests/
│   ├── test_encoders.py
│   ├── test_fusion.py
│   └── test_pipeline.py
│
├── notebooks/
│   ├── exploration.ipynb
│   └── visualization.ipynb
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## 10. 기술 스택

| 영역 | 권장 도구 | 버전 |
|-----|----------|-----|
| 프레임워크 | PyTorch | 2.0+ |
| Transformer | HuggingFace Transformers | 4.30+ |
| Vision | timm, torchvision | 최신 |
| Audio | torchaudio, librosa | 최신 |
| 분산 학습 | PyTorch Lightning, DeepSpeed | 최신 |
| 데이터 로딩 | WebDataset, FFCV | 최신 |
| 저장 | h5py, pyarrow, safetensors | 최신 |
| 실험 관리 | Weights & Biases, MLflow | 최신 |
| 코드 품질 | black, ruff, mypy | 최신 |

---

## 11. 학습 데이터셋

본 파이프라인에서 활용 가능한 모달리티별 공개 데이터셋을 정리합니다.

### 11.1 데이터셋 플랫폼

| 플랫폼 | URL | 특징 |
|-------|-----|------|
| **Hugging Face Datasets** | https://huggingface.co/datasets | 가장 활발, 다양한 모달리티, Python API 제공 |
| **Kaggle** | https://kaggle.com/datasets | 경진대회 데이터, 커뮤니티 활발 |
| **Papers With Code** | https://paperswithcode.com/datasets | 논문 연계 벤치마크 데이터셋 |
| **Google Dataset Search** | https://datasetsearch.research.google.com | 메타 검색 엔진 |
| **AWS Open Data** | https://registry.opendata.aws | 대용량 데이터, S3 호스팅 |

### 11.2 Vision (이미지/비디오) 데이터셋

| 데이터셋 | 규모 | 내용 | 라이선스 | 링크 |
|---------|-----|------|---------|------|
| **COCO** | 33만 이미지 | 객체 탐지, 세그멘테이션, 캡션 | CC-BY 4.0 | https://cocodataset.org |
| **ImageNet** | 1,400만 이미지 | 1000 클래스 분류 | 연구용 | https://image-net.org |
| **LAION-5B** | 50억 이미지-텍스트 쌍 | 웹 크롤링 멀티모달 | 연구용 | https://laion.ai/blog/laion-5b |
| **Open Images V7** | 900만 이미지 | 객체 탐지, 관계 | CC-BY 4.0 | https://storage.googleapis.com/openimages |
| **Conceptual Captions** | 1,200만 쌍 | 이미지-캡션 | 연구용 | https://ai.google.com/research/ConceptualCaptions |
| **WebVid-10M** | 1,000만 비디오 | 비디오-텍스트 | 연구용 | https://m-bain.github.io/webvid-dataset |
| **Kinetics-700** | 65만 비디오 | 행동 인식 700 클래스 | 연구용 | https://deepmind.com/research/open-source/kinetics |
| **SA-1B** | 11억 마스크 | 세그멘테이션 (SAM용) | Apache 2.0 | https://segment-anything.com |

**COCO 데이터셋 상세:**
```
구성:
├── train2017/          # 118,287 이미지
├── val2017/            # 5,000 이미지
├── test2017/           # 40,670 이미지
└── annotations/
    ├── captions_*.json       # 이미지당 5개 캡션
    ├── instances_*.json      # 객체 바운딩 박스 + 세그멘테이션
    └── keypoints_*.json      # 인체 키포인트

다운로드:
- 이미지: ~25GB
- 어노테이션: ~1GB
```

**활용 코드:**
```python
from datasets import load_dataset

# Hugging Face에서 COCO 로드
coco = load_dataset("detection-datasets/coco", split="train")

# 또는 pycocotools 사용
from pycocotools.coco import COCO
coco_api = COCO('annotations/instances_train2017.json')
```

### 11.3 Audio (음성/사운드) 데이터셋

| 데이터셋 | 규모 | 내용 | 라이선스 | 링크 |
|---------|-----|------|---------|------|
| **LibriSpeech** | 1,000시간 | 영어 오디오북 음성 | CC-BY 4.0 | https://openslr.org/12 |
| **Common Voice** | 19,000+시간 | 다국어 음성 (한국어 포함) | CC0 | https://commonvoice.mozilla.org |
| **AudioSet** | 200만 클립 | 632 오디오 이벤트 클래스 | CC-BY 4.0 | https://research.google.com/audioset |
| **VoxCeleb 1&2** | 100만+ 발화 | 화자 인식 (7,000+ 화자) | 연구용 | https://www.robots.ox.ac.uk/~vgg/data/voxceleb |
| **AudioCaps** | 46,000 클립 | 오디오 캡셔닝 | 연구용 | https://audiocaps.github.io |
| **MUSAN** | 음악/음성/노이즈 | 데이터 증강용 | 다양 | https://openslr.org/17 |
| **GigaSpeech** | 10,000시간 | 영어 ASR | Apache 2.0 | https://github.com/SpeechColab/GigaSpeech |
| **MLS** | 50,000+시간 | 다국어 LibriSpeech | CC-BY 4.0 | https://openslr.org/94 |

**LibriSpeech 데이터셋 상세:**
```
구성:
├── train-clean-100/    # 100시간 깨끗한 음성
├── train-clean-360/    # 360시간 깨끗한 음성
├── train-other-500/    # 500시간 (노이즈 포함)
├── dev-clean/          # 검증셋 (깨끗)
├── dev-other/          # 검증셋 (노이즈)
├── test-clean/         # 테스트셋 (깨끗)
└── test-other/         # 테스트셋 (노이즈)

형식:
- 오디오: 16kHz FLAC
- 전사: .txt 파일

다운로드:
- train-clean-100: ~6GB
- 전체: ~60GB
```

**활용 코드:**
```python
from datasets import load_dataset
import torchaudio

# Hugging Face에서 로드
librispeech = load_dataset("librispeech_asr", "clean", split="train.100")

# 샘플 접근
sample = librispeech[0]
audio = sample["audio"]["array"]  # numpy array
sr = sample["audio"]["sampling_rate"]  # 16000
text = sample["text"]  # 전사 텍스트
```

### 11.4 Text (자연어) 데이터셋

| 데이터셋 | 규모 | 내용 | 라이선스 | 링크 |
|---------|-----|------|---------|------|
| **SQuAD 2.0** | 150,000 QA | 독해 기반 질의응답 | CC-BY-SA 4.0 | https://rajpurkar.github.io/SQuAD-explorer |
| **The Pile** | 800GB | LLM 사전학습용 텍스트 | MIT | https://pile.eleuther.ai |
| **Common Crawl** | 페타바이트급 | 웹 크롤링 텍스트 | 다양 | https://commoncrawl.org |
| **Wikipedia Dumps** | 다국어 | 위키피디아 전체 덤프 | CC-BY-SA | https://dumps.wikimedia.org |
| **RedPajama** | 1.2조 토큰 | LLaMA 재현용 | Apache 2.0 | https://github.com/togethercomputer/RedPajama-Data |
| **OpenWebText2** | 69GB | Reddit 링크 기반 | 연구용 | https://openwebtext2.readthedocs.io |
| **C4** | 750GB | 정제된 Common Crawl | ODC-BY | https://huggingface.co/datasets/c4 |

**SQuAD 데이터셋 상세:**
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
```

### 11.5 멀티모달 통합 데이터셋

여러 모달리티가 이미 정렬되어 있는 데이터셋:

| 데이터셋 | 모달리티 | 규모 | 용도 | 링크 |
|---------|---------|-----|------|------|
| **LAION-400M** | 이미지+텍스트 | 4억 쌍 | Vision-Language 사전학습 | https://laion.ai |
| **CC3M/CC12M** | 이미지+텍스트 | 300만/1,200만 | 이미지 캡셔닝 | https://github.com/google-research-datasets/conceptual-12m |
| **VQA v2** | 이미지+질문+답변 | 120만 QA | Visual QA | https://visualqa.org |
| **HowTo100M** | 비디오+텍스트+음성 | 1억 클립 | 비디오 이해 | https://www.di.ens.fr/willow/research/howto100m |
| **AVSpeech** | 비디오+오디오 | 29만 클립 | Audio-Visual 학습 | https://looking-to-listen.github.io/avspeech |
| **VATEX** | 비디오+텍스트 (중영) | 41,000 비디오 | 다국어 비디오 캡션 | https://eric-xw.github.io/vatex-website |
| **Spoken-SQuAD** | 음성+텍스트 | 37,000 QA | Spoken QA | https://github.com/chiahsuan156/Spoken-SQuAD |
| **MELD** | 비디오+오디오+텍스트 | 13,000 발화 | 감정 인식 | https://affective-meld.github.io |

### 11.6 Tactile / Robotics 데이터셋

| 데이터셋 | 내용 | 센서 유형 | 링크 |
|---------|-----|----------|------|
| **RoboNet** | 로봇 조작 비디오 | RGB + 로봇 상태 | https://bair.berkeley.edu/blog/2019/11/26/robonet |
| **MIME** | 인간-로봇 상호작용 | 모션캡처 + 힘 | https://sites.google.com/view/mimedataset |
| **ContactDB** | 접촉 패턴 | 열화상 + 깊이 | https://contactdb.cc.gatech.edu |
| **YCB Benchmarks** | 물체 조작 | RGB-D + 포즈 | https://rse-lab.cs.washington.edu/projects/posecnn |
| **Touch and Go** | 촉각 탐색 | DIGIT 센서 | https://touch-and-go.github.io |
| **ObjectFolder 2.0** | 시각+촉각+청각 | 멀티센서 | https://ai.stanford.edu/~rhgao/objectfolder |
| **DexYCB** | 손 조작 | RGB-D + 손 포즈 | https://dex-ycb.github.io |

### 11.7 한국어/아시아 특화 데이터셋

| 데이터셋 | 모달리티 | 내용 | 링크 |
|---------|---------|-----|------|
| **AI Hub** | 다중 | 한국어 음성, 이미지, 텍스트 종합 | https://aihub.or.kr |
| **KsponSpeech** | 음성 | 1,000시간 한국어 자발화 | AI Hub |
| **모두의 말뭉치** | 텍스트 | 한국어 텍스트 코퍼스 | https://corpus.korean.go.kr |
| **KorQuAD 2.0** | 텍스트 | 한국어 QA (SQuAD 스타일) | https://korquad.github.io |
| **KLUE** | 텍스트 | 한국어 NLU 벤치마크 8종 | https://klue-benchmark.com |
| **Zeroth-Korean** | 음성 | 51시간 한국어 음성 | https://openslr.org/40 |
| **AIHub 멀티모달** | 다중 | 한국어 이미지 캡션, VQA | AI Hub |

**AI Hub 주요 데이터셋:**
```
음성:
- 한국어 음성 (1,000시간+)
- 감정 음성
- 노인/아동 음성
- 다화자 음성

이미지:
- 한국 음식 이미지
- 한국어 OCR
- 의료 영상

텍스트:
- 법률/의료 말뭉치
- 대화 데이터
- 기계독해 데이터

멀티모달:
- 멀티모달 감성분석
- 이미지-텍스트 쌍
```

### 11.8 데이터셋 선택 가이드

**프로젝트 단계별 권장:**

| 단계 | 권장 데이터셋 | 이유 |
|-----|-------------|------|
| **MVP/프로토타입** | COCO + LibriSpeech + SQuAD | 소규모, 잘 정제됨, 접근 용이 |
| **중규모 학습** | CC12M + Common Voice + Wikipedia | 균형잡힌 규모 |
| **대규모 사전학습** | LAION-5B + GigaSpeech + The Pile | 최대 규모 |
| **한국어 특화** | AI Hub 종합 활용 | 한국어 품질 보장 |

**모달리티 조합별 권장:**

| 조합 | 권장 데이터셋 |
|-----|-------------|
| Vision + Text | COCO Captions, CC12M, VQA v2 |
| Audio + Text | LibriSpeech, Common Voice, AudioCaps |
| Vision + Audio | AVSpeech, VGGSound |
| Vision + Audio + Text | HowTo100M, MELD |
| 전체 모달리티 | ObjectFolder (제한적 규모) |

### 11.9 라이선스 주의사항

| 라이선스 | 상업적 사용 | 주요 데이터셋 |
|---------|-----------|-------------|
| **CC0** | ✅ 완전 자유 | Common Voice (일부) |
| **CC-BY** | ✅ 출처 표기 | COCO, AudioSet, LibriSpeech |
| **CC-BY-SA** | ✅ 동일 조건 | Wikipedia, SQuAD |
| **CC-BY-NC** | ❌ 비상업만 | 일부 학술 데이터셋 |
| **연구용** | ❌ 연구만 | ImageNet, LAION (URL만) |
| **Apache 2.0** | ✅ 자유 | GigaSpeech, RedPajama |

### 11.10 데이터 로딩 유틸리티

```python
# 통합 데이터 로더 예시
from datasets import load_dataset
import torchaudio
from PIL import Image

class MultimodalDataLoader:
    """여러 데이터셋을 통합 로드하는 유틸리티"""
    
    DATASET_CONFIGS = {
        'coco': {
            'hf_name': 'detection-datasets/coco',
            'modalities': ['vision', 'text'],
            'vision_key': 'image',
            'text_key': 'captions'
        },
        'librispeech': {
            'hf_name': 'librispeech_asr',
            'modalities': ['audio', 'text'],
            'audio_key': 'audio',
            'text_key': 'text'
        },
        'squad': {
            'hf_name': 'squad_v2',
            'modalities': ['text'],
            'text_keys': ['context', 'question', 'answers']
        }
    }
    
    @classmethod
    def load(cls, dataset_name: str, split: str = 'train', **kwargs):
        config = cls.DATASET_CONFIGS[dataset_name]
        return load_dataset(config['hf_name'], split=split, **kwargs)
```

---

## 12. 구현 로드맵

### Phase 1: MVP (2주)
- [x] Vision + Text 2개 모달리티 파이프라인
- [x] 기본 Cross-Attention 융합
- [x] HDF5 출력

### Phase 2: 확장 (2주)
- [ ] Audio 모달리티 추가
- [ ] Missing modality 처리
- [ ] 품질 필터링

### Phase 3: 최적화 (1주)
- [ ] Mixed Precision 적용
- [ ] Flash Attention 통합
- [ ] 분산 학습 지원

### Phase 4: 검증 (1주)
- [ ] Downstream task 평가
- [ ] 벤치마크 데이터셋 테스트
- [ ] 문서화 완료

---

## 13. 참고 자료

### 관련 연구
- CLIP (Radford et al., 2021) - Vision-Language 정렬
- ImageBind (Girdhar et al., 2023) - 6개 모달리티 통합
- Perceiver (Jaegle et al., 2021) - 확장 가능한 융합 아키텍처
- Flamingo (Alayrac et al., 2022) - Cross-attention 기반 멀티모달

### 유용한 링크
- HuggingFace Transformers: https://huggingface.co/docs/transformers
- PyTorch Lightning: https://lightning.ai/docs/pytorch/stable/
- Weights & Biases: https://docs.wandb.ai/

---

> **문서 작성**: Claude  
> **최종 수정**: 2025년 12월

