# -*- coding: utf-8 -*-
"""
Wrapper Ren'Py — protège tokens/balises, anti-coupure, correction 's et espaces, post-vérif via .backup.
- Reconnaît aussi les tokens SANS double underscore: RENPY_CODE_003, RENPY_EMPTY01, RENPY_NARRATOR, RENPY_ASTER, etc.
- Canonicalise toutes les formes vers __RENPY_TAG__ ou __RENPY_TAG_999__ (sortie).
- Corrige immédiatement (même sans .backup) :
    • possessif anglais 's après un span balisé ou un token
    • espaces entre mot↔token, token↔mot, mot↔{balise}, {/balise}↔mot
- Conserve le log de progression (~X.X l/s)
"""

import os
import re
import time
from typing import List, Tuple
import importlib

_core = importlib.import_module("traducteur_renpy")
CoreTraducteur = getattr(_core, "TraducteurRenPy")


class TraducteurRenPy(CoreTraducteur):
    # Options
    enable_fr_grammar: bool = False
    auto_install_languagetool: bool = False


    # ==========================
    # Limites sûres & helpers
    # ==========================
    def _true_model_max_len(self) -> int:
        m = getattr(self, "model", None)
        cfg = getattr(m, "config", None) if m is not None else None
        for key in ("max_position_embeddings", "max_length", "max_source_positions", "n_positions"):
            v = getattr(cfg, key, None) if cfg is not None else None
            if isinstance(v, int) and 0 < v < 100000:
                return int(v)
        return 1024

    def _tokenizer_model_max_len(self) -> int:
        ml = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(ml, int) and 0 < ml < 100000:
            return int(ml)
        return 10**9

    def _safe_input_len(self) -> int:
        true_max = self._true_model_max_len()
        tok_max  = self._tokenizer_model_max_len()
        safe = min(true_max - 96, tok_max)  # petite marge
        return max(256, int(safe))

    def _tok_len(self, s: str) -> int:
        try:
            from transformers.utils import logging as hf_logging
            old = hf_logging.get_verbosity()
            hf_logging.set_verbosity_error()
            try:
                ids = self.tokenizer(s, return_tensors="pt", truncation=False)["input_ids"]
                return int(ids.shape[1])
            finally:
                hf_logging.set_verbosity(old)
        except Exception:
            # fallback approx
            return max(1, len(s) // 3 + 4)

    # ==========================
    # Normalisation légère
    # ==========================
    _TOKEN_ANY = r"(?:__)?RENPY_[A-Z]+(?:_?[0-9]+)?(?:__)?"
    _TOKEN_ANY_RE = re.compile(_TOKEN_ANY)

    def _normalize_segment(self, s: str) -> str:
        s = (s or "").replace("\r", " ").replace("\n", " ")
        s = re.sub(r"\{\s*color\s*=\s*([0-9A-Fa-f]{6})\s*\}", r"{color=\1}", s)
        s = re.sub(r"\{\s*/\s*color\s*\}", r"{/color}", s)
        return s

    def _strip_possessive_s_simple(self, s: str) -> str:
        # {span}'s -> {span}␠
        s = re.sub(r"(\{[A-Za-z_]+[^}]*\}[^{}]*\{/[A-Za-z_]+\})\s*['’]s\b", r"\1 ", s)
        TOKEN = r"RENPY_[A-Z]+(?:_?[0-9]+)?"
        # TOKEN 's -> TOKEN␠
        s = re.sub(rf"({TOKEN})\s*['’]s\b", r"\1 ", s)
        # TOKEN <mot> TOKEN 's -> TOKEN<mot>TOKEN␠  (cas: RENPY_CODE_006SophieRENPY_CODE_003's)
        s = re.sub(rf"({TOKEN})([^\s{{}}]+)({TOKEN})\s*['’]s\b", r"\1\2\3 ", s)
        return s


    def _tokenize_RENPY(self, text: str):
        token_pat = re.compile(r"RENPY_[A-Z]+(?:_?[0-9]+)?")
        pos = 0
        chunks, tokens = [], []
        for m in token_pat.finditer(text):
            if m.start() > pos:
                chunks.append(text[pos:m.start()])
            else:
                chunks.append("")
            tokens.append(m.group(0))
            pos = m.end()
        chunks.append(text[pos:])
        return chunks, tokens



    def _normalize_list(self, arr: List[str]) -> List[str]:
        return [self._normalize_segment(x) for x in (arr or [])]

    # ==========================
    # Éléments protégés (gelés)
    # ==========================
    _PROTECTED = re.compile(
        r"(?:__)?RENPY_[A-Z]+(?:_?[0-9]+)?(?:__)?"
        r"|"
        r"\{[a-zA-Z_]+(?:=[^}]*)?\}"
        r"|"
        r"\{/[a-zA-Z_]+\}"
        r"|"
        r"\[[^\]]+\]"            # ⬅️ nouvelles variables protégées: [mcname], [points], etc.
    )

    def _split_protected(self, text: str) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        last = 0
        for m in self._PROTECTED.finditer(text):
            if m.start() > last:
                out.append(("text", text[last:m.start()]))
            out.append(("token", m.group(0)))
            last = m.end()
        if last < len(text):
            out.append(("text", text[last:]))
        return out or [("text", "")]

    # ==========================
    # Découpe sûre du texte
    # ==========================
    def _split_text_to_safe_chunks(self, text: str) -> List[str]:
        text = text or ""
        if not text:
            return [""]
        SAFE = self._safe_input_len()
        if len(text) <= 180:
            return [text]
        if self._tok_len(text) <= SAFE:
            return [text]

        parts = re.split(r'(?<=[.!?…])\s+', text)
        chunks: List[str] = []
        cur = ""
        for s in parts:
            proposal = (cur + " " + s).strip() if cur else s
            if self._tok_len(proposal) > SAFE:
                if cur:
                    chunks.append(cur)
                    cur = s
                else:
                    chunks += self._split_greedy_by_words(s, SAFE)
                    cur = ""
            else:
                cur = proposal
        if cur:
            chunks.append(cur)
        if chunks:
            return chunks
        return self._split_greedy_by_words(text, SAFE)

    def _split_greedy_by_words(self, s: str, SAFE: int) -> List[str]:
        words = s.split()
        out: List[str] = []
        cur = ""
        for w in words:
            proposal = (cur + " " + w).strip() if cur else w
            if self._tok_len(proposal) > SAFE:
                if cur:
                    out.append(cur)
                    cur = w
                else:
                    out += self._split_by_chars(w, SAFE)
                    cur = ""
            else:
                cur = proposal
        if cur:
            out.append(cur)
        return out or [s]

    def _split_by_chars(self, s: str, SAFE: int) -> List[str]:
        out: List[str] = []
        i = 0
        step = max(120, min(800, len(s)))
        while i < len(s):
            sub = s[i:i+step]
            while self._tok_len(sub) > SAFE and step > 40:
                step = max(40, step // 2)
                sub = s[i:i+step]
            if not sub:
                break
            out.append(sub)
            i += len(sub)
        return out or [s]

    # ==========================
    # Fix tiret & espaces balises
    # ==========================
    def _enforce_leading_dash_if_in_source(self, core_src: str, core_out: str) -> str:
        if not core_src:
            return core_out or ""
        SP = r"[ \t\u00A0\u2000-\u200A\u202F\u205F\u3000]"
        m = re.match(rf"^({SP}*[–—-]{SP}*)", core_src)
        if not m:
            return core_out or ""
        prefix_src = m.group(1)
        out = core_out or ""
        idx_dash = None
        for ch in ('–', '—', '-'):
            k = out.find(ch)
            if k != -1:
                idx_dash = k if idx_dash is None else min(idx_dash, k)
        if idx_dash is None:
            return prefix_src + out.lstrip()
        rest = out[idx_dash+1:]
        rest = re.sub(rf"^{SP}+", "", rest)
        return prefix_src + rest



    # ==========================
    # Post-vérif avec .backup
    # ==========================
    def _post_verify_with_backup(self, translated_path: str):
        backup_path = translated_path + ".backup"
        if not os.path.exists(backup_path):
            return
        try:
            with open(translated_path, "r", encoding="utf-8", errors="ignore") as f:
                out_lines = f.readlines()
            with open(backup_path, "r", encoding="utf-8", errors="ignore") as f:
                bak_lines = f.readlines()
        except Exception:
            return

        n = min(len(out_lines), len(bak_lines))
        fixed = []

        for i in range(n):
            out_line = out_lines[i]
            bak_line = bak_lines[i]

            # conserver fin de ligne
            newline = "\r\n" if out_line.endswith("\r\n") else ("\n" if out_line.endswith("\n") else "")
            out_body = out_line.rstrip("\r\n")
            bak_body = bak_line.rstrip("\r\n")

            # (a) SPAN's -> un espace (toujours)
            #     ex: {color=...}Sophie{/color}'s -> {color=...}Sophie{/color}␠
            out_body = re.sub(r"(\{[A-Za-z_]+[^}]*\}[^{}]*\{/[A-Za-z_]+\})\s*['’]s\b", r"\1 ", out_body)

            # (b) TOKEN's -> on suit le backup :
            #     si le backup contient RENPY_XXX's, alors on force "RENPY_XXX " dans la sortie
            TOKEN_RE = r"RENPY_[A-Z]+(?:_?[0-9]+)?"
            poss_tokens = { m.group(1) for m in re.finditer(rf"({TOKEN_RE})['’]s\b", bak_body) }
            for tok in poss_tokens:
                # 1) si la sortie a encore "'s", on le remplace par un espace
                out_body = re.sub(rf"({re.escape(tok)})\s*['’]s\b", r"\1 ", out_body)
                # 2) sinon, s'il n'y a PAS d'espace après le token et qu'un mot suit, on insère 1 espace
                out_body = re.sub(rf"({re.escape(tok)})(?!\s)(?=[A-Za-zÀ-ÖØ-öø-ÿ])", r"\1 ", out_body)

            # (b2) VARIABLE's -> on suit le backup :
            #      si le backup contient [var]'s, alors on force "[var] " dans la sortie
            VAR_RE = r"\[[^\]]+\]"
            poss_vars = { m.group(1) for m in re.finditer(rf"({VAR_RE})['’]s\b", bak_body) }
            for var in poss_vars:
                # 1) si la sortie a encore "'s", on le remplace par un espace
                out_body = re.sub(rf"({re.escape(var)})\s*['’]s\b", r"\1 ", out_body)
                # 2) sinon, si aucune espace après la variable et qu'une lettre suit, insérer 1 espace
                out_body = re.sub(rf"({re.escape(var)})(?!\s)(?=[A-Za-zÀ-ÖØ-öø-ÿ])", r"\1 ", out_body)

            # (c) Espaces entre tokens calqués sur le backup, SANS jamais supprimer du texte (ex: 'Sophie','Helen')
            out_chunks, out_tokens = self._tokenize_RENPY(out_body)
            bak_chunks, bak_tokens = self._tokenize_RENPY(bak_body)

            if out_tokens and len(out_tokens) == len(bak_tokens):
                rebuilt = []
                rebuilt.append(out_chunks[0])  # chunk avant 1er token (on le garde tel quel)

                for k, tok in enumerate(out_tokens):
                    rebuilt.append(tok)

                    if k + 1 < len(out_tokens):
                        out_gap = out_chunks[k + 1]
                        bak_gap = bak_chunks[k + 1]

                        if re.search(r"\S", out_gap):
                            # du TEXTE entre tokens (ex: 'Sophie') -> on ne change rien
                            rebuilt.append(out_gap)
                        else:
                            # gap blanc -> on reproduit l'espace (ou pas) du backup
                            rebuilt.append(" " if re.search(r"\s", bak_gap) else "")
                    else:
                        # après le dernier token -> garder le dernier chunk de sortie tel quel
                        tail = out_chunks[-1] if len(out_chunks) == len(out_tokens) + 1 else ""
                        rebuilt.append(tail)

                out_body = "".join(rebuilt)


            # (d) Restaure la ponctuation perdue après éléments protégés si le backup l'avait
            PUNCT = r"[,:;.!?…]"

            # 1) Après variables [nom]
            for m in re.finditer(r"(\[[^\[\]\s]+\])\s*(" + PUNCT + r")", bak_body):
                base, punct = m.group(1), m.group(2)
                # si la sortie n'a pas déjà cette ponctuation juste après, on l'insère + espace
                out_body = re.sub(
                    r"(" + re.escape(base) + r")\s*(?!"+re.escape(punct)+r")",
                    r"\1" + punct + " ",
                    out_body,
                    count=1
                )
                # normalise espace après la ponctuation
                out_body = re.sub(
                    r"(" + re.escape(base + punct) + r")\s*",
                    r"\1 ",
                    out_body,
                    count=1
                )

            # 2) Après tokens RENPY_XXX
            for m in re.finditer(r"(RENPY_[A-Z]+(?:_?[0-9]+)?)\s*(" + PUNCT + r")", bak_body):
                base, punct = m.group(1), m.group(2)
                out_body = re.sub(
                    r"(" + re.escape(base) + r")\s*(?!"+re.escape(punct)+r")",
                    r"\1" + punct + " ",
                    out_body,
                    count=1
                )
                out_body = re.sub(
                    r"(" + re.escape(base + punct) + r")\s*",
                    r"\1 ",
                    out_body,
                    count=1
                )

            # 3) Après balises fermantes {/xxx}
            for m in re.finditer(r"(\{/[A-Za-z_]+\})\s*(" + PUNCT + r")", bak_body):
                base, punct = m.group(1), m.group(2)
                out_body = re.sub(
                    r"(" + re.escape(base) + r")\s*(?!"+re.escape(punct)+r")",
                    r"\1" + punct + " ",
                    out_body,
                    count=1
                )
                out_body = re.sub(
                    r"(" + re.escape(base + punct) + r")\s*",
                    r"\1 ",
                    out_body,
                    count=1
                )



            # ré-écrire la ligne
            fixed.append(out_body + newline)

        # lignes restantes (s'il y en a)
        for i in range(n, len(out_lines)):
            line = out_lines[i]
            newline = "\r\n" if line.endswith("\r\n") else ("\n" if line.endswith("\n") else "")
            body = self._strip_possessive_s_simple(line.rstrip("\r\n"))
            fixed.append(body + newline)

        try:
            with open(translated_path, "w", encoding="utf-8", newline="") as f:
                f.writelines(fixed)
        except Exception:
            pass



    # ==========================
    # API fichiers
    # ==========================
    def traduire_fichier_sans_coupure_interne(self, src_path: str, dst_path: str, log_every_s: float = 0.8):
        with open(src_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
            raw = f.read()

        lines = raw.splitlines(keepends=True)
        if not lines:
            os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
            with open(dst_path, "w", encoding="utf-8", newline="") as f:
                f.write("")
            return

        total_lines = len(lines)
        # Détection auto de la langue (échantillon)
        self._ensure_src_lang_from_sample([l.rstrip("\r\n") for l in lines])

        # Découpage en segments protégés/texte avec conservation des espaces de bord
        structures: List[List[Tuple]] = []
        for l in lines:
            body = l.rstrip("\r\n")
            segments = []
            for kind, payload in self._split_protected(body):
                if kind == "text":
                    m = re.match(r'^(\s*)(.*?)(\s*)$', payload, flags=re.DOTALL)
                    if m:
                        lws, core, rws = m.group(1), m.group(2), m.group(3)
                    else:
                        lws, core, rws = "", payload, ""
                    segments.append(("text", core, lws, rws))
                else:
                    segments.append(("token", payload))
            structures.append(segments)

        # Suffixes exacts de fin de ligne
        suffixes = []
        for l in lines:
            if l.endswith("\r\n"): suffixes.append("\r\n")
            elif l.endswith("\n"):  suffixes.append("\n")
            elif l.endswith("\r"):  suffixes.append("\r")
            else:                    suffixes.append("")

        # Aplatir → traduire seulement les "core" textes (ignorer vides/sans lettres)
        text_chunks: List[str] = []
        seg_chunk_map: List[Tuple[int,int,int,str,str]] = []  # (line_idx, seg_idx, nb_chunks, lws, rws)
        for li, segs in enumerate(structures):
            for si, seg in enumerate(segs):
                if seg[0] == "text":
                    _, core, lws, rws = seg
                    has_letters = bool(re.search(r'[A-Za-zÀ-ÖØ-öø-ÿ]', core))
                    if core.strip() and has_letters:
                        chs = self._split_text_to_safe_chunks(core)
                        text_chunks.extend(chs)
                        seg_chunk_map.append((li, si, len(chs), lws, rws))
                    else:
                        seg_chunk_map.append((li, si, 0, lws, rws))

        # Traduction (par batch) avec log l/s
        results_text: List[str] = []
        Bp = 32
        t0 = time.time()
        last_log = 0.0
        done_lines = 0

        def _count_chunks_for_line(segs) -> int:
            c = 0
            for s in segs:
                if s[0] == "text":
                    core = s[1]
                    if core.strip() and re.search(r'[A-Za-zÀ-ÖØ-öø-ÿ]', core):
                        c += len(self._split_text_to_safe_chunks(core))
            return c

        line_text_chunk_totals = [_count_chunks_for_line(segs) for segs in structures]
        cum_line_text_chunks = []
        acc = 0
        for v in line_text_chunk_totals:
            acc += v
            cum_line_text_chunks.append(acc)

        for i in range(0, len(text_chunks), Bp):
            sub = text_chunks[i:i+Bp]
            try:
                trad = self.traduire_batch(sub)
            except Exception:
                trad = []
                for s in sub:
                    try: trad.append(self.traduire_texte_simple(s))
                    except Exception: trad.append(s)
            trad = self._normalize_list(trad)
            results_text.extend(trad)

            translated_chunks = len(results_text)
            while done_lines < total_lines and translated_chunks >= cum_line_text_chunks[done_lines]:
                done_lines += 1

            now = time.time()
            if (now - last_log) >= log_every_s:
                rate = done_lines / max(1e-6, now - t0)
                print(f"   ⏳ Progression: {done_lines}/{total_lines} lignes (~{rate:.1f} l/s)")
                last_log = now

        # recomposition 1→1
        out_lines: List[str] = []
        text_idx = 0
        seginfo = { (li, si): (nb, lws, rws) for (li, si, nb, lws, rws) in seg_chunk_map }

        for li, segs in enumerate(structures):
            rebuilt_parts: List[str] = []
            for si, seg in enumerate(segs):
                if seg[0] == "text":
                    _, core, lws0, rws0 = seg
                    nb, lws, rws = seginfo.get((li, si), (0, lws0, rws0))
                    if nb > 0:
                        core_out = " ".join(self._normalize_list(results_text[text_idx:text_idx+nb]))
                        text_idx += nb
                    else:
                        # ⬅️ SEGMENT NON TRADUIT (pas de lettres) : on garde le texte original tel quel
                        core_out = core
                    core_out = core_out.strip()
                    core_out = self._enforce_leading_dash_if_in_source(core, core_out)
                    rebuilt_parts.append(lws + core_out + rws)
                else:
                    rebuilt_parts.append(seg[1])


            joined = "".join(rebuilt_parts)
            norm = self._normalize_segment(joined)   # seulement color={...} et {/color}
            final_line = norm + suffixes[li]
            out_lines.append(final_line)

        # écriture
        os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
        with open(dst_path, "w", encoding="utf-8", newline="") as f:
            f.write("".join(out_lines))

        # post-vérification avec le backup
        self._post_verify_with_backup(dst_path)

    # compat
    def traduire_fichier_rapide(self, src_path: str, dst_path: str):
        return self.traduire_fichier_sans_coupure_interne(src_path, dst_path)
