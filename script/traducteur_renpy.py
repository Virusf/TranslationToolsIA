# -*- coding: utf-8 -*-
"""
Traducteur Ren'Py — Core optimisé
- NUM_BEAMS = 1 (perf)
- FP16 sur GPU + SDPA/Flash Attention (PyTorch 2.x)
- TF32 activé (Ada/Ampere) si dispo
- Garde VRAM stricte: headroom 2.0 GB
- Encodage tronqué à une longueur sûre (marge tokens spéciaux)
- Batch adaptatif avec réduction en cas d'OOM / dépassement limite
"""

import os, re, time
from typing import List, Optional

# --- Anti-warning PyTorch SDP déprécié ---
import warnings
warnings.filterwarnings(
    "ignore",
    message="`torch.backends.cuda.sdp_kernel\\(\\)` is deprecated",
    category=FutureWarning,
)

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

### CHANGEMENT 1: Importer PeftModel ###
try:
    from peft import PeftModel
except ImportError:
    print("⚠️  La bibliothèque 'peft' n'est pas installée. Le chargement de LoRA est désactivé. Faites 'pip install peft' pour l'activer.")
    PeftModel = None
### FIN CHANGEMENT 1 ###

# --- Langdetect (auto-install soft) ---
try:
    from langdetect import detect, detect_langs
except Exception:
    try:
        import subprocess, sys as _sys
        subprocess.check_call([_sys.executable, "-m", "pip", "install", "langdetect"])
        from langdetect import detect, detect_langs
    except Exception:
        detect = None
        detect_langs = None


CONFIG = {
    "BATCH_SIZE": 64,
    "MIN_BATCH": 4,
    "MAX_BATCH": 128,
    "AUTO_BATCH": True,
    "VRAM_HEADROOM_GB": 2.0,   # marge VRAM stricte
    "NUM_BEAMS": 3,            # vitesse
}


# ISO -> NLLB
ISO2NLLB = {
    # --- Européennes ---
    "en": "eng_Latn",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "pt-br": "por_Latn",
    "nl": "nld_Latn",
    "af": "afr_Latn",
    "sv": "swe_Latn",
    "da": "dan_Latn",
    "no": "nob_Latn",
    "fi": "fin_Latn",
    "is": "isl_Latn",
    "pl": "pol_Latn",
    "cs": "ces_Latn",
    "sk": "slk_Latn",
    "sl": "slv_Latn",
    "hr": "hrv_Latn",
    "sr": "srp_Cyrl",
    "bs": "bos_Latn",
    "mk": "mkd_Cyrl",
    "bg": "bul_Cyrl",
    "ro": "ron_Latn",
    "hu": "hun_Latn",
    "el": "ell_Grek",
    "et": "est_Latn",
    "lv": "lav_Latn",
    "lt": "lit_Latn",
    "uk": "ukr_Cyrl",
    "ru": "rus_Cyrl",

    # --- Moyen-Orient ---
    "tr": "tur_Latn",
    "az": "azj_Latn",
    "hy": "hye_Armn",
    "ka": "kat_Geor",
    "fa": "pes_Arab",
    "ar": "arb_Arab",
    "he": "heb_Hebr",
    "ku": "kmr_Latn",

    # --- Asie du Sud ---
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "ur": "urd_Arab",
    "pa": "pan_Guru",
    "gu": "guj_Gujr",
    "ml": "mal_Mlym",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "kn": "kan_Knda",
    "si": "sin_Sinh",
    "ne": "npi_Deva",

    # --- Asie du Sud-Est ---
    "id": "ind_Latn",
    "ms": "zsm_Latn",
    "jv": "jav_Latn",
    "su": "sun_Latn",
    "th": "tha_Thai",
    "km": "khm_Khmr",
    "lo": "lao_Laoo",
    "my": "mya_Mymr",
    "vi": "vie_Latn",
    "tl": "tgl_Latn",

    # --- Asie de l'Est ---
    "zh": "zho_Hans",     # défaut simplifié
    "zh-cn": "zho_Hans",
    "zh-tw": "zho_Hant",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "mn": "khk_Cyrl",

    # --- Afrique ---
    "sw": "swh_Latn",
    "am": "amh_Ethi",
    "so": "som_Latn",
    "yo": "yor_Latn",
    "ig": "ibo_Latn",
    "ha": "hau_Latn",
    "zu": "zul_Latn",
    "xh": "xho_Latn",
    "st": "sot_Latn",

    # --- Autres ---
    "eo": "epo_Latn",
    "la": "lat_Latn",
}




def _try(x, fn, default=None):
    try:
        return fn(x)
    except Exception:
        return default

class TraducteurRenPy:
    def __init__(self, model_path, src_lang: str = "eng_Latn", tgt_lang: str = "fra_Latn", lora_path: Optional[str] = None):
        # Perf knobs globaux
        try:
            torch.set_float32_matmul_precision("high")  # TF32
        except Exception:
            pass

        # SDPA/Flash: active seulement si la nouvelle API existe (sinon rien)
        try:
            from torch.nn.attention import sdpa_kernel
            sdpa_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
        except Exception:
            pass


        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        print("🚀 Chargement du modèle (perf) ...", flush=True)
        t_load0 = time.time()
        print("   ⏳ Préparation (détection repo/local)...", flush=True)
        is_repo = isinstance(model_path, str) and re.match(r"^[\w.-]+/[\w.-]+$", model_path or "")
        print(f"   📦 Source: {'HuggingFace' if is_repo else 'Local files'}", flush=True)


        ### CHANGEMENT 3: Logique de chargement du modèle et du LoRA ###
        print("   ⏳ Téléchargement/lecture du modèle de base...", flush=True)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=not is_repo)
        
        if lora_path and PeftModel:
            print(f"   🎨 Application du LoRA depuis : {lora_path}", flush=True)
            try:
                is_lora_repo = isinstance(lora_path, str) and re.match(r"^[\w.-]+/[\w.-]+$", lora_path or "")
                self.model = PeftModel.from_pretrained(base_model, lora_path, local_files_only=not is_lora_repo)
                print("   ✅ LoRA appliqué avec succès.", flush=True)
            except Exception as e:
                print(f"❌ Échec de l'application du LoRA : {e}", flush=True)
                print("   -> Utilisation du modèle de base seul.", flush=True)
                self.model = base_model
        else:
            self.model = base_model
        ### FIN CHANGEMENT 3 ###


        # print("   ⏳ Téléchargement/lecture du modèle...", flush=True)
        # self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=not is_repo)
        # print(f"   ✅ Modèle chargé ({time.time()-t_load0:.1f}s)", flush=True)

        print("   ⏳ Chargement du tokenizer (rapide si déjà en cache)...", flush=True)
        try:
            t_tok = time.time()
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=not is_repo)
            print(f"   ✅ Tokenizer prêt ({time.time()-t_tok:.1f}s, fast=True)", flush=True)
        except Exception:
            t_tok = time.time()
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, local_files_only=not is_repo)
            print(f"   ✅ Tokenizer prêt ({time.time()-t_tok:.1f}s, fast=False)", flush=True)

        if torch.cuda.is_available():
            print("   🎮 Passage sur GPU + FP16 ...", flush=True)
            self.device = "cuda"
            try:
                self.model = self.model.to(dtype=torch.float16)
            except Exception:
                self.model = self.model.half()
            self.model = self.model.to(self.device).eval()
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"🎮 GPU: VRAM totale ~ {vram_total:.1f} GB")
            self._vram_total = float(vram_total)
        else:
            print("   ⚙️ Mode CPU (plus lent)", flush=True)
            self.device = "cpu"
            self.model = self.model.eval()
            self._vram_total = None

        print("✅ Modèle prêt!", flush=True)
        


        # langues
        if hasattr(self.tokenizer, "src_lang") and self.src_lang != "auto":
            self.tokenizer.src_lang = self.src_lang

        self._prefix_mode = False
        self._tgt_lang_id = self._resolve_lang_id(self.tgt_lang)
        if self._tgt_lang_id is None:
            self._prefix_mode = True

        # limite d'entrée sûre
        self._SAFE_INPUT_LEN = self._compute_safe_input_len()
        print(f"🔒 Longueur d'entrée sûre (tokens): {self._SAFE_INPUT_LEN}")

        # batch / vram guard
        self._cur_B = int(CONFIG.get("BATCH_SIZE", 64))
        self._auto = bool(CONFIG.get("AUTO_BATCH", True))
        headroom = float(CONFIG.get("VRAM_HEADROOM_GB", 2.0))
        if torch.cuda.is_available():
            self._vram_limit = max(0.0, self._vram_total - headroom)
            print(f"🛡️  VRAM headroom: {headroom:.1f} GB → limite d'utilisation ~ {self._vram_limit:.2f} GB")
            if self._auto:
                self._cur_B = max(CONFIG.get("MIN_BATCH", 4), min(self._cur_B, CONFIG.get("MAX_BATCH", 128)))
            torch.cuda.empty_cache()
        else:
            self._vram_limit = None
            print(f"⚙️  CPU mode | Beams: {CONFIG.get('NUM_BEAMS', 1)}")

    # ---- helpers limites ----
    def _true_model_max_len(self) -> int:
        cfg = getattr(self.model, "config", None)
        for key in ("max_position_embeddings", "max_length", "max_source_positions", "n_positions"):
            v = getattr(cfg, key, None) if cfg is not None else None
            if isinstance(v, int) and 0 < v < 100000:
                return int(v)
        return 1024

    def _tokenizer_model_max_len(self) -> Optional[int]:
        ml = _try(self.tokenizer, lambda t: t.model_max_length, None)
        if isinstance(ml, int) and 0 < ml < 100000:
            return int(ml)
        return None

    def _compute_safe_input_len(self) -> int:
        true_max = self._true_model_max_len()
        tok_max = self._tokenizer_model_max_len() or 10**9
        safe = min(true_max - 96, tok_max)  # marge plus large
        return max(256, int(safe))

    def _resolve_lang_id(self, code: str):
        tok = self.tokenizer
        # direct convert
        tid = _try(code, tok.convert_tokens_to_ids, None)
        unk = getattr(tok, "unk_token_id", None)
        if isinstance(tid, int) and tid is not None and (unk is None or tid != unk):
            return int(tid)
        # fallback
        if hasattr(tok, "get_lang_id"):
            tid = _try(code, tok.get_lang_id, None)
            if isinstance(tid, int):
                return int(tid)
        # specials
        try:
            specials = {t: tok.convert_tokens_to_ids(t) for t in getattr(tok, "all_special_tokens", [])}
            if code in specials and specials[code] is not None:
                return int(specials[code])
        except Exception:
            pass
        return None

    # def _make_gen_kwargs(self):
    #     kw = dict(
    #         max_new_tokens=64,
    #         num_beams=CONFIG.get("NUM_BEAMS", 1),
    #         do_sample=False,
    #         use_cache=True,
    #     )
    #     if not self._prefix_mode and self._tgt_lang_id is not None:
    #         kw["forced_bos_token_id"] = self._tgt_lang_id
    #     if getattr(self.tokenizer, "pad_token_id", None) is not None:
    #         kw["pad_token_id"] = self.tokenizer.pad_token_id
    #     return kw

    def _make_gen_kwargs(self):
        kw = dict(
            max_new_tokens=256,  # ⬅️ augmenter la marge (64 → 256)
            num_beams=CONFIG.get("NUM_BEAMS", 1),
            do_sample=False,
            use_cache=True,
        )
        if not self._prefix_mode and self._tgt_lang_id is not None:
            kw["forced_bos_token_id"] = self._tgt_lang_id
        if getattr(self.tokenizer, "pad_token_id", None) is not None:
            kw["pad_token_id"] = self.tokenizer.pad_token_id
        return kw


    # encodage sûr
    def _encode_with_safe_len(self, texts: List[str]):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=int(self._SAFE_INPUT_LEN),
            return_tensors="pt"
        )

    # ---- VRAM guard ----
    def _over_limit(self, margin: float = 0.98) -> bool:
        if not torch.cuda.is_available(): return False
        alloc = torch.cuda.memory_allocated() / 1e9
        reserv = torch.cuda.memory_reserved() / 1e9
        return (alloc > self._vram_limit * margin) or (reserv > self._vram_limit * margin)

    def _guarded_generate_batch(self, sub: List[str]) -> List[str]:
        Bmin = CONFIG.get("MIN_BATCH", 4)
        B = len(sub)
        out_all = []
        i = 0
        while i < len(sub):
            cur = sub[i:i+B]
            try:
                enc = self._encode_with_safe_len(cur)
                enc = {k: v.to(self.device) for k, v in enc.items()}
                if self._over_limit():
                    raise torch.cuda.OutOfMemoryError("pre-generate over VRAM limit")
                with torch.no_grad():
                    out = self.model.generate(**enc, **self._make_gen_kwargs())
                out_all.extend(self.tokenizer.batch_decode(out, skip_special_tokens=True))
                i += len(cur)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if B <= Bmin:
                    # fallback élément par élément
                    for s in cur:
                        try:
                            out_all.append(self._guarded_generate_single(s))
                        except Exception:
                            out_all.append(s)
                    i += len(cur)
                else:
                    B = max(Bmin, B // 2)
                    print(f"⚠️  VRAM guard → réduction du sous-batch à {B}")
            except Exception:
                # fallback élément par élément
                for s in cur:
                    try:
                        out_all.append(self._guarded_generate_single(s))
                    except Exception:
                        out_all.append(s)
                i += len(cur)
        return out_all

    def _guarded_generate_single(self, text: str) -> str:
        enc = self._encode_with_safe_len([text])
        enc = {k: v.to(self.device) for k, v in enc.items()}
        if self._over_limit():
            torch.cuda.empty_cache()
        with torch.no_grad():
            out = self.model.generate(**enc, **self._make_gen_kwargs())
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    # ---- API ----
    def traduire_texte_simple(self, texte: str) -> str:
        if not texte: return texte
        txt = f"{self.tgt_lang} {texte}" if self._prefix_mode else texte
        return self._guarded_generate_single(txt)

    def traduire_batch(self, textes: List[str]) -> List[str]:
        if not textes: return []
        batch = [f"{self.tgt_lang} {t}" for t in textes] if self._prefix_mode else list(textes)
        out_all: List[str] = []
        B = int(self._cur_B or 64)
        B = max(CONFIG.get("MIN_BATCH", 4), min(B, CONFIG.get("MAX_BATCH", 128)))
        i = 0
        while i < len(batch):
            sub = batch[i:i+B]
            out = self._guarded_generate_batch(sub)
            out_all.extend(out)
            i += len(sub)
        return out_all

    # --- heuristiques de langue (léger) ---
    def _extract_dialog_bits(self, lines: List[str], max_chars: int = 2000) -> str:
        _DLG_RE = re.compile(r'(?:"([^"]+)"|\'([^\']+)\')')
        _SCRIPT_HEAD_RE = re.compile(r'^(label|translate|define|default|init|screen|transform|image|python:|menu\s*:?)\b')
        buf, total = [], 0
        for ln in lines:
            s = (ln or "").strip()
            if not s or _SCRIPT_HEAD_RE.match(s) or s.startswith("#"):
                continue
            for m in _DLG_RE.finditer(s):
                piece = m.group(1) or m.group(2) or ""
                if piece:
                    buf.append(piece)
                    total += len(piece)
                    if total >= max_chars:
                        return " ".join(buf)
        if not buf:
            for ln in lines:
                s = (ln or "").strip()
                if s and not _SCRIPT_HEAD_RE.match(s) and not s.startswith("#"):
                    buf.append(s)
                    total += len(s)
                    if total >= max_chars:
                        break
        return " ".join(buf)


    def _ensure_src_lang_from_sample(self, lines: List[str]):
        """
        Détecte la langue majoritaire du fichier (fenêtrage + vote) avec langdetect.
        Si indisponible/échec -> fallback FR/EN simplifié.
        """
        if self.src_lang != "auto":
            # Forcée par l'UI → on respecte.
            try:
                if hasattr(self.tokenizer, "src_lang"):
                    self.tokenizer.src_lang = self.src_lang
            except Exception:
                pass
            return

        # 1) Construire un gros échantillon textuel à partir des dialogues
        sample = self._extract_dialog_bits(lines, max_chars=20000)
        sample_low = sample.lower()

        detected_iso = None

        # 2) Détection robuste (fenêtres) si langdetect dispo
        if detect and detect_langs:
            # Fenêtrage pour votes majoritaires (plus fiable sur fichiers hétérogènes)
            window = 800
            step = 800
            votes = {}
            for i in range(0, len(sample), step):
                chunk = sample[i:i+window].strip()
                if len(chunk) < 30:
                    continue
                try:
                    # detect_langs → "xx:prob,yy:prob..."
                    langs = detect_langs(chunk)
                    if langs:
                        top = sorted(langs, key=lambda x: x.prob, reverse=True)[0]
                        iso = top.lang.lower()
                        votes[iso] = votes.get(iso, 0) + 1
                except Exception:
                    continue

            if votes:
                detected_iso = max(votes.items(), key=lambda kv: kv[1])[0]

        # 3) Fallback simple FR/EN si rien détecté
        if not detected_iso:
            iso = "en"
            if any(w in sample_low for w in [" le ", " la ", " et ", " je ", " tu "]):
                iso = "fr"
            detected_iso = iso

        # 4) Mapping ISO -> NLLB (avec défaut anglais)
        nllb = ISO2NLLB.get(detected_iso, "eng_Latn")
        self.src_lang = nllb
        try:
            if hasattr(self.tokenizer, "src_lang"):
                self.tokenizer.src_lang = nllb
        except Exception:
            pass
        print(f"🧭 Langue source détectée (majoritaire): {detected_iso} → {nllb}")
