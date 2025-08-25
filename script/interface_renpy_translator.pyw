# -*- coding: utf-8 -*-
"""
Interface Ren'Py Translator — ordre ajusté + bloc langues complet
1) Dossier des fichiers à traduire
2) Parcourir récursivement les sous-dossiers
3) Dossier de sortie (optionnel)
Puis: Chemin du modèle, Langues (avec autodétection), Actions, Log
"""

import os
import sys
import re
import threading
import subprocess
from tkinter import messagebox
import json

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import traceback
import shutil
import time
import queue
import importlib

APP_TITLE = "Interface Ren'Py Translator"

def _is_hf_repo_id(s: str) -> bool:
    return bool(re.match(r"^[\w.-]+/[\w.-]+$", s or ""))

class _TeeStream:
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def write(self, s):
        if self.a:
            try: self.a.write(s)
            except Exception: pass
        if self.b:
            try: self.b.write(s)
            except Exception: pass
    def flush(self):
        for t in (self.a, self.b):
            if t:
                try: t.flush()
                except Exception: pass

class _TextBoxWriter:
    def __init__(self, q):
        self.q = q
        self._buffer = ""
    def write(self, s):
        self._buffer += s
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self.q.put(line + "\n")
    def flush(self):
        if self._buffer:
            self.q.put(self._buffer)
            self._buffer = ""

class InterfaceRenPyTranslator:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("650x540")
        
        self.config_file = "config.json"

        self.batch_log_enabled = tk.BooleanVar(value=False)
        self.batch_log_path    = tk.StringVar(value="")
        self.batch_log_clean   = tk.BooleanVar(value=False)

        # Thème sombre simple
        self.root.configure(bg="#3c3c3c")
        self.root.option_add("*Foreground", "white")
        self.root.option_add("*Background", "#3c3c3c")
        self.root.option_add("*Button.Background", "#3c3f41")
        self.root.option_add("*Button.Foreground", "white")
        self.root.option_add("*Entry.Background", "#3c3f41")
        self.root.option_add("*Entry.Foreground", "white")
        self.root.option_add("*Label.Background", "#3c3c3c")

        # State
        self.dossier_jeu = tk.StringVar()
        self.model_path = tk.StringVar(value=self.get_default_model_path())
        self.dossier_sortie = tk.StringVar(value="")
        self.recursif = tk.BooleanVar(value=True)
        self._worker = None
        self._stop_flag = False
        self.src_lang = tk.StringVar(value="auto")
        self.tgt_lang = tk.StringVar(value="fra_Latn")
        self.autodetect_src = tk.BooleanVar(value=True)
        self.grammar_fr = tk.BooleanVar(value=False)

        # Log infra
        self._log_queue = queue.Queue()
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr

        self._build_ui()
        self._drain_queue()
        
        self.load_settings()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.log("Sélectionne le dossier puis clique sur 'Lancer la traduction'.")

    def get_default_model_path(self):
        """
        Détermine le chemin du modèle par défaut avec une logique de priorité :
        1. Le modèle par défaut spécifique s'il existe localement.
        2. Le premier autre modèle trouvé dans le dossier ./models.
        3. L'identifiant Hugging Face comme solution de repli.
        """
        specific_local_model = "./models/hub/models--virusf--nllb-renpy-rory-v3"
        models_dir = "./models"
        hf_repo_id = "virusf/nllb-renpy-rory-v3"

        if os.path.isdir(specific_local_model):
            print(f"Modèle par défaut local trouvé : {specific_local_model}")
            return specific_local_model

        if os.path.isdir(models_dir):
            try:
                for item in os.listdir(models_dir):
                    full_path = os.path.join(models_dir, item)
                    if os.path.isdir(full_path):
                        print(f"Autre modèle local trouvé : {full_path}")
                        return full_path
            except OSError as e:
                print(f"Erreur en lisant le dossier models : {e}")

        print(f"Aucun modèle local trouvé. Utilisation de l'ID Hugging Face : {hf_repo_id}")
        print("Le modèle sera téléchargé dans ./models lors de la première traduction.")
        return hf_repo_id

    def save_settings(self):
        """Sauvegarde les paramètres actuels dans un fichier JSON."""
        settings = {
            "dossier_jeu": self.dossier_jeu.get(),
            "model_path": self.model_path.get(),
            "dossier_sortie": self.dossier_sortie.get(),
            "recursif": self.recursif.get(),
            "src_lang": self.src_lang.get(),
            "tgt_lang": self.tgt_lang.get(),
            "autodetect_src": self.autodetect_src.get(),
            "grammar_fr": self.grammar_fr.get(),
        }
        try:
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=4)
        except Exception as e:
            print(f"Error saving settings: {e}")

    def load_settings(self):
        """Charge les paramètres depuis un fichier JSON s'il existe."""
        if not os.path.exists(self.config_file):
            return  # Pas de fichier de config, on garde les valeurs par défaut
        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                settings = json.load(f)

            # Charger tous les autres paramètres
            self.dossier_jeu.set(settings.get("dossier_jeu", ""))
            self.dossier_sortie.set(settings.get("dossier_sortie", ""))
            self.recursif.set(settings.get("recursif", True))
            self.src_lang.set(settings.get("src_lang", "auto"))
            self.tgt_lang.set(settings.get("tgt_lang", "fra_Latn"))
            self.autodetect_src.set(settings.get("autodetect_src", True))
            self.grammar_fr.set(settings.get("grammar_fr", False))

            # Gérer le chemin du modèle : s'il est vide ou absent, on applique la détection par défaut.
            saved_model_path = settings.get("model_path")
            if saved_model_path:  # N'est vrai que si la chaîne n'est pas vide
                self.model_path.set(saved_model_path)
            else:
                self.model_path.set(self.get_default_model_path())

            # Mettre à jour l'état de l'interface après le chargement
            self._toggle_src_field()

            self.log("⚙️ Paramètres chargés.")
        except Exception as e:
            self.log(f"⚠️ Fichier de configuration corrompu ou illisible : {e}")

    def on_closing(self):
        """Appelée lorsque l'utilisateur ferme la fenêtre."""
        self.save_settings()
        self.root.destroy()
        
    def _build_ui(self):
        # 1) Dossier des fichiers à traduire
        frame_game = tk.Frame(self.root)
        frame_game.pack(fill="x", padx=10, pady=(8, 2))
        tk.Label(frame_game, text="Dossier des fichiers à traduire :").pack(side="left")
        tk.Entry(frame_game, textvariable=self.dossier_jeu, width=60).pack(side="left", padx=5)
        tk.Button(frame_game, text="Parcourir", command=self.choisir_dossier_jeu).pack(side="left")

        # 2) Parcourir récursivement les sous-dossiers
        frame_opts = tk.Frame(self.root)
        frame_opts.pack(fill="x", padx=10, pady=(2, 2))
        tk.Checkbutton(frame_opts, text="Parcourir récursivement les sous-dossiers", variable=self.recursif).pack(anchor="w")

        # 4) Chemin du modèle
        frame_model = tk.Frame(self.root)
        frame_model.pack(fill="x", padx=10, pady=5)
        tk.Label(frame_model, text="Chemin du modèle :").pack(side="left")
        tk.Entry(frame_model, textvariable=self.model_path, width=65).pack(side="left", padx=5)
        tk.Button(frame_model, text="Parcourir", command=self.choisir_modele).pack(side="left")

        # 5) Langues (avec autodétection + ligne source/cible)
        frame_lang = tk.Frame(self.root)
        frame_lang.pack(fill="x", padx=10, pady=5)
        tk.Checkbutton(
            frame_lang,
            text="Détecter automatiquement la langue source",
            variable=self.autodetect_src,
            command=self._toggle_src_field
        ).pack(anchor="w")
        line2 = tk.Frame(frame_lang)
        line2.pack(fill="x", pady=(6, 0))

        tk.Label(line2, text="Langue source (code NLLB) :").pack(side="left")
        self.src_entry = tk.Entry(
            line2, textvariable=self.src_lang, width=12,
            state="disabled" if self.autodetect_src.get() else "normal"
        )
        self.src_entry.pack(side="left", padx=5)
        tk.Label(line2, text="→ Cible :").pack(side="left", padx=10)
        tk.Entry(line2, textvariable=self.tgt_lang, width=12).pack(side="left", padx=5)

        # 6) Actions
        frame_actions = tk.Frame(self.root)
        frame_actions.pack(fill="x", padx=10, pady=8)
        tk.Button(frame_actions, text="Lancer la traduction", command=self.lancer_traduction).pack(side="left")
        tk.Button(frame_actions, text="Effacer le log", command=self.clear_log).pack(side="left", padx=10)

        # 7) Log
        self.logbox = scrolledtext.ScrolledText(self.root, height=22, wrap="word", font=("Consolas", 10))
        self.logbox.pack(fill="both", expand=True, padx=10, pady=10)

    # ---------- Helpers ----------
    def _toggle_src_field(self):
        self.src_entry.configure(state="disabled" if self.autodetect_src.get() else "normal")
        if self.autodetect_src.get():
            self.src_lang.set("auto")

    def log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.logbox.insert("end", f"[{ts}] {msg}\n")
        self.logbox.see("end")

    def clear_log(self):
        self.logbox.delete("1.0", "end")

    def _drain_queue(self):
        try:
            while True:
                line = self._log_queue.get_nowait()
                self.logbox.insert("end", line)
                self.logbox.see("end")
        except queue.Empty:
            pass
        self.root.after(50, self._drain_queue)

    def choisir_dossier_jeu(self):
        d = filedialog.askdirectory()
        if d: self.dossier_jeu.set(d)

    def choisir_modele(self):
        d = filedialog.askdirectory()
        if d: self.model_path.set(d)

    def choisir_dossier_sortie(self):
        d = filedialog.askdirectory()
        if d: self.dossier_sortie.set(d)

    def _collect_rpy_files(self, base_dir, recursive=True):
        paths = []
        if recursive:
            for r, _, files in os.walk(base_dir):
                for f in files:
                    if f.lower().endswith((".rpy",".txt")):
                        paths.append(os.path.join(r, f))
        else:
            for f in os.listdir(base_dir):
                p = os.path.join(base_dir, f)
                if os.path.isfile(p) and f.lower().endswith((".rpy",".txt")):
                    paths.append(p)
        return paths

    def _choose_batch_log_file(self):
        from tkinter.filedialog import asksaveasfilename
        path = asksaveasfilename(defaultextension=".log", filetypes=[("Log", "*.log"), ("Texte", "*.txt"), ("Tous", "*.*")])
        if path:
            self.batch_log_path.set(path)

    def lancer_traduction(self):
        if getattr(self, '_worker', None) and self._worker.is_alive():
            messagebox.showinfo("Patiente", "Une traduction est déjà en cours.")
            return

        # Forcer le téléchargement des modèles dans ./models en utilisant la nouvelle variable
        os.environ["HF_HOME"] = os.path.abspath("./models")
        
        chemin = self.dossier_jeu.get().strip()
        modele = self.model_path.get().strip()
        sortie = self.dossier_sortie.get().strip()
        recurse = self.recursif.get()
        src = self.src_lang.get().strip()
        tgt = self.tgt_lang.get().strip()

        os.environ["RENPY_BATCHLOG_ENABLE"] = "1" if self.batch_log_enabled.get() else "0"
        os.environ["RENPY_BATCHLOG_PATH"]   = self.batch_log_path.get().strip()
        os.environ["RENPY_BATCHLOG_CLEAN"]  = "1" if self.batch_log_clean.get() else "0" 

        if not chemin or not os.path.isdir(chemin):
            messagebox.showwarning("Dossier du jeu", "Merci de sélectionner un dossier valide.")
            return

        if not modele or (not os.path.isdir(modele) and not _is_hf_repo_id(modele)):
            messagebox.showwarning("Chemin du modèle",
                "Indique un dossier local (ex: ./models/nllb) ou un ID HF (ex: virusf/nllb-renpy-rory-v3).")
            return

        self._stop_flag = False
        self._worker = threading.Thread(
            target=self._job_traduction, 
            args=(chemin, modele, sortie, recurse, src, tgt),
            daemon=True
        )
        self._worker.start()

    def _job_traduction(self, chemin, modele, sortie, recurse, src_lang, tgt_lang):
        ui_writer = _TextBoxWriter(self._log_queue)
        self._old_out, self._old_err = sys.stdout, sys.stderr
        sys.stdout = _TeeStream(self._old_out, ui_writer)
        sys.stderr = _TeeStream(self._old_err, ui_writer)

        try:
            print(f"📁 Dossier : {chemin}")
            print(f"🧠 Modèle : {modele}")
            print(f"🌐 Langues : {src_lang} → {tgt_lang}")
            print("📂 Mode :", "Récursif" if recurse else "Ce dossier seulement")
            if sortie:
                print(f"📤 Dossier de sortie : {sortie}")
            else:
                print("✍️ Écrasement des fichiers (backup auto .backup)")

            rpy_files = self._collect_rpy_files(chemin, recursive=recurse)
            if not rpy_files:
                print("ℹ️ Aucun fichier .rpy/.txt trouvé.")
                return

            print(f"🔎 Fichiers trouvés : {len(rpy_files)}")

            try:
                mod = importlib.import_module('traducteur_renpy_wrapper')
                TraducteurRenPy = getattr(mod, 'TraducteurRenPy')
            except Exception:
                mod = importlib.import_module('traducteur_renpy')
                TraducteurRenPy = getattr(mod, 'TraducteurRenPy')

            try:
                traducteur = TraducteurRenPy(modele, src_lang=src_lang, tgt_lang=tgt_lang)
                try:
                    setattr(traducteur, "auto_install_languagetool", False)
                except Exception:
                    pass
                try:
                    setattr(traducteur, "enable_fr_grammar", bool(self.grammar_fr.get()))
                except Exception:
                    pass
            except Exception as e:
                print("❌ Échec init traducteur:", e)
                print(traceback.format_exc())
                return

            translate_file = getattr(traducteur, "traduire_fichier_sans_coupure_interne", None)
            if not callable(translate_file):
                translate_file = getattr(traducteur, "traduire_fichier_rapide", None)
            if not callable(translate_file):
                print("ℹ️  Ni traduire_fichier_sans_coupure_interne ni traduire_fichier_rapide — fallback core-only")
                def _fallback_translate_file(_src, _dst):
                    import re, os, time
                    TOKEN = re.compile(r"(?:__)?RENPY_[A-Z]+(?:_?[0-9]+)?(?:__)?|\{[a-zA-Z_]+(?:=[^}]*)?\}|\{/[a-zA-Z_]+\}")
                    LTR  = r"A-Za-zÀ-ÖØ-öø-ÿ0-9«“\"'’"
                    t0 = time.time()
                    last_log = t0
                    with open(_src, "r", encoding="utf-8", errors="ignore", newline="") as f:
                        raw = f.read()
                    out_lines = []
                    lines = raw.splitlines(keepends=True)
                    for i, line in enumerate(lines, 1):
                        body = line.rstrip("\r\n")
                        rebuilt, last = [], 0
                        for m in TOKEN.finditer(body):
                            if m.start() > last:
                                seg = body[last:m.start()]
                                rebuilt.append(traducteur.traduire_texte_simple(seg) if seg.strip() else seg)
                            rebuilt.append(m.group(0))
                            last = m.end()
                        if last < len(body):
                            seg = body[last:]
                            rebuilt.append(traducteur.traduire_texte_simple(seg) if seg.strip() else seg)
                        s = "".join(rebuilt)
                        s = re.sub(r"(\{[A-Za-z_]+[^}]*\}[^{}]*\{/[A-Za-z_]+\})\s*['’]s\b", r"\1 ", s)
                        s = re.sub(rf"(\{{/[A-Za-z_]+\}})([{LTR}])", r"\1 \2", s)
                        s = re.sub(rf"([{LTR}])(\{{[A-Za-z_]+(?:=[^}}]*)?\}})", r"\1 \2", s)
                        s = re.sub(r"([A-Za-zÀ-ÖØ-öø-ÿ])(__RENPY_[A-Z]+(?:_[0-9]+)?__)", r"\1 \2", s)
                        s = re.sub(r"(__RENPY_[A-Z]+(?:_[0-9]+)?__)([A-Za-zÀ-ÖØ-öø-ÿ])", r"\1 \2", s)
                        out_lines.append(s + ("\n" if line.endswith("\n") else ""))
                        now = time.time()
                        if now - last_log >= 0.8:
                            rate = i / max(1e-6, now - t0)
                            print(f"   ⏳ Progression (fallback): {i}/{len(lines)} lignes (~{rate:.1f} l/s)")
                            last_log = now
                    os.makedirs(os.path.dirname(_dst) or ".", exist_ok=True)
                    with open(_dst, "w", encoding="utf-8", newline="") as f:
                        f.write("".join(out_lines))
                translate_file = _fallback_translate_file

            ok = failed = 0
            for idx, src in enumerate(rpy_files, 1):
                rel = os.path.relpath(src, chemin)
                try:
                    if sortie:
                        dest = os.path.join(sortie, rel)
                        os.makedirs(os.path.dirname(dest), exist_ok=True)
                        print(f"➡️  [{idx}/{len(rpy_files)}] Traduit vers : {rel}")
                        try:
                            shutil.copy2(src, dest + ".backup")
                        except Exception as e:
                            print(f"⚠️ Backup (out_dir) raté pour {rel} : {e}")
                        translate_file(src, dest)
                    else:
                        backup_path = src + ".backup"
                        if not os.path.exists(backup_path):
                            try: shutil.copy2(src, backup_path)
                            except Exception as e: print(f"⚠️ Backup raté pour {rel} : {e}")
                        print(f"✏️  [{idx}/{len(rpy_files)}] Écrase : {rel} (backup: {os.path.basename(backup_path)})")
                        translate_file(src, src)
                    print(f"✅ Fini : {rel}")
                    ok += 1
                except Exception as e:
                    failed += 1
                    print(f"❌ Erreur sur {rel} : {e}")
                    print(traceback.format_exc())

            print(f"✅ Terminé : {ok} fichiers réussis, {failed} en échec.")
        finally:
            sys.stdout = self._old_out
            sys.stderr = self._old_err

def main():
    root = tk.Tk()
    app = InterfaceRenPyTranslator(root)
    root.mainloop()

if __name__ == "__main__":
    main()