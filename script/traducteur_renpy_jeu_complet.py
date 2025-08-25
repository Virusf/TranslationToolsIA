
# -*- coding: utf-8 -*-
"""
Traduction Ren'Py (jeu complet) — conserve strictement la structure des fichiers
Processus:
- Parcourt un dossier
- Pour chaque .rpy/.txt, traduit ligne par ligne (1→1) en préservant les fins de ligne
"""
import os
import shutil
from typing import List
import importlib

def _collect_files(base_dir: str, recursive=True) -> List[str]:
    paths = []
    if recursive:
        for r, _, files in os.walk(base_dir):
            for f in files:
                if f.endswith(".rpy") or f.endswith(".txt"):
                    paths.append(os.path.join(r, f))
    else:
        for f in os.listdir(base_dir):
            p = os.path.join(base_dir, f)
            if os.path.isfile(p) and (p.endswith(".rpy") or p.endswith(".txt")):
                paths.append(p)
    return paths

def translate_game_folder(folder: str, model_path: str, src_lang: str, tgt_lang: str, out_dir: str = ""):
    # Importer le wrapper qui préserve la structure
    mod = importlib.import_module("traducteur_renpy_wrapper")
    TraducteurRenPy = getattr(mod, "TraducteurRenPy")

    trad = TraducteurRenPy(model_path, src_lang=src_lang, tgt_lang=tgt_lang)

    files = _collect_files(folder, recursive=True)
    for p in files:
        rel = os.path.relpath(p, folder)
        if out_dir:
            dst = os.path.join(out_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            print(f"➡️  Traduit vers : {rel}")
            trad.traduire_fichier_rapide(p, dst)
        else:
            backup = p + ".backup"
            if not os.path.exists(backup):
                try:
                    shutil.copy2(p, backup)
                except Exception as e:
                    print(f"⚠️ Impossible de créer le backup pour {rel} : {e}")
            print(f"✏️  Écrase : {rel} (backup: {os.path.basename(backup)})")
            trad.traduire_fichier_rapide(p, p)
