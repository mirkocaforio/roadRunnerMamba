import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionBranch(nn.Module):
    """
    Branch ausiliario leggero per il disentanglement condizione/texture.

    Motivazione specifica per RSCD wet vs water:
    - La FG mask con k=0.05 cattura bene la STRUTTURA della superficie
      (grana asfalto, striature cemento) ma sopprime i token che
      codificano la CONDIZIONE (riflessi diffusi, lucentezza globale).
    - wet e water differiscono principalmente per intensita' della
      condizione di bagnato, non per struttura — quindi la FG mask
      li rende difficili da separare.
    - Questo branch opera sul segnale GLOBALE (average pooling su tutti
      i token, inclusi quelli soppressi dalla mask) e impara a stimare
      la condizione ambientale come task ausiliario.

    Come si integra con il branch principale:
    - Il branch principale (SSM + FG mask) produce feature di texture
    - Questo branch produce un vettore di condizione
    - I due vengono combinati tramite cross-modulation prima della head:
        texture_final = texture * (1 + condition_gate)
      dove condition_gate e' una proiezione learnable del vettore condizione.
      Questo permette alla condizione di "riscalare" le feature di texture
      in modo selettivo per canale.

    Args:
        d_model:        dimensione feature in input (= num_features del backbone)
        num_conditions: numero di condizioni ambientali da predire
                        per simple-RSCD: 5 (dry, wet, water, snow, ice)
                        per RSCD completo: 6 (+ ice separato)
        hidden_dim:     dimensione layer nascosto del branch (default d_model//4)
        dropout:        dropout nel branch ausiliario
    """

    def __init__(
        self,
        d_model: int,
        num_conditions: int = 6,
        hidden_dim: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_conditions = num_conditions
        hidden_dim = hidden_dim or d_model // 4

        # Encoder condizione: feature globali -> vettore condizione
        # Leggero di proposito: non deve rubare capacita' al branch principale
        self.condition_encoder = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Head ausiliaria per predire la condizione (usata solo in training)
        self.condition_head = nn.Linear(hidden_dim, num_conditions)

        # Gate: proietta il vettore condizione in uno scale per le feature
        # di texture — permette alla condizione di modulare selettivamente
        # i canali del branch principale
        # Output: [B, d_model] con valori in (-1, 1) via tanh
        self.condition_gate = nn.Sequential(
            nn.Linear(hidden_dim, d_model),
            nn.Tanh(),  # gate in (-1, 1): puo' amplificare o sopprimere
        )

        self._init_weights()

    def _init_weights(self):
        # Inizializza il gate a zero cosi' all'inizio del training
        # condition_gate produce 0 -> texture_final = texture * 1 (identita')
        # Il branch parte "trasparente" e impara gradualmente
        nn.init.zeros_(self.condition_gate[0].weight)
        nn.init.zeros_(self.condition_gate[0].bias)

    def forward(self, texture_features: torch.Tensor):
        """
        Args:
            texture_features: [B, d_model] — output del backbone dopo avgpool
                              rappresenta le feature di texture globali

        Returns:
            modulated_features: [B, d_model] — feature modulate dalla condizione
            condition_logits:   [B, num_conditions] — logits per la loss ausiliaria
                                None in inferenza se non servono
        """
        # Encoding della condizione dal segnale globale
        cond_feat = self.condition_encoder(texture_features)  # [B, hidden_dim]

        # Gate: quanto ogni canale di texture e' influenzato dalla condizione
        gate = self.condition_gate(cond_feat)                 # [B, d_model]

        # Cross-modulation: la condizione riscala le feature di texture
        # texture_final = texture * (1 + gate)
        # - gate=0 (inizio training): identita', nessun effetto
        # - gate>0: amplifica i canali correlati alla condizione
        # - gate<0: sopprime i canali non discriminativi per la condizione
        modulated = texture_features * (1.0 + gate)

        return modulated