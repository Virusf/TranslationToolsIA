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

APP_TITLE = "Ren'Py Translator Interface"

# Codes NLLB usuels → Libellés lisibles
NLLB_CODES = {
    # Europe (latin)
    "eng_Latn":"English", 
    "fra_Latn":"Français", 
    "spa_Latn":"Español",
    "deu_Latn":"Deutsch", 
    "ita_Latn":"Italiano", 
    "nld_Latn":"Nederlands", 
    "por_Latn":"Português", "swe_Latn":"Svenska", "dan_Latn":"Dansk", "fin_Latn":"Suomi", "isl_Latn":"Íslenska",
    "pol_Latn":"Polski", "ces_Latn":"Čeština", "slk_Latn":"Slovenčina",
    "slv_Latn":"Slovenščina", "hrv_Latn":"Hrvatski", "bos_Latn":"Bosanski",
    "ron_Latn":"Română", "hun_Latn":"Magyar", "est_Latn":"Eesti", "lav_Latn":"Latviešu",
    "lit_Latn":"Lietuvių",
    # Europe (autres scripts)
    "ell_Grek":"Ελληνικά",
    "rus_Cyrl":"Русский", 
    "ukr_Cyrl":"Українська", 
    "bul_Cyrl":"Български", 
    "mkd_Cyrl":"Македонски",
    "srp_Cyrl":"Српски",
    # Moyen-Orient
    "tur_Latn":"Türkçe", "azj_Latn":"Azərbaycanca", "hye_Armn":"Հայերեն", "kat_Geor":"ქართული",
    "arb_Arab":"العربية", "pes_Arab":"فارسی", "heb_Hebr":"עברית", "kmr_Latn":"Kurdî (Kurmanji)",
    # Asie du Sud
    "hin_Deva":"हिन्दी", "ben_Beng":"বাংলা", "urd_Arab":"اردو", "pan_Guru":"ਪੰਜਾਬੀ",
    "guj_Gujr":"ગુજરાતી", "mal_Mlym":"മലയാളം", "tam_Taml":"தமிழ்", "tel_Telu":"తెలుగు",
    "kan_Knda":"ಕನ್ನಡ", "sin_Sinh":"සිංහල", "npi_Deva":"नेपाली",
    # Asie du Sud-Est
    "ind_Latn":"Bahasa Indonesia", "zsm_Latn":"Bahasa Melayu", "jav_Latn":"Basa Jawa",
    "sun_Latn":"Basa Sunda", "tha_Thai":"ไทย", "khm_Khmr":"ខ្មែរ", "lao_Laoo":"ລາວ",
    "mya_Mymr":"မြန်မာ", "vie_Latn":"Tiếng Việt", "tgl_Latn":"Tagalog",
    # Asie de l’Est
    "jpn_Jpan":"日本語", 
    "zho_Hans":"中文（简体）", 
    "zho_Hant":"中文（繁體）", 
    "kor_Hang":"한국어",
    "khk_Cyrl":"Монгол",
    # Afrique
    "swh_Latn":"Kiswahili", "amh_Ethi":"አማርኛ", "som_Latn":"Af-Soomaali",
    "yor_Latn":"Yorùbá", "ibo_Latn":"Asụsụ Igbo", "hau_Latn":"Hausa",
    "zul_Latn":"isiZulu", "xho_Latn":"isiXhosa", "sot_Latn":"Sesotho",
    # Divers
    "epo_Latn":"Esperanto", "lat_Latn":"Latina",
}


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
        self.root.geometry("600x550")
        
        self.config_file = "config.json"

        self.batch_log_enabled = tk.BooleanVar(value=False)
        self.batch_log_path    = tk.StringVar(value="")
        self.batch_log_clean   = tk.BooleanVar(value=False)

        # --- Thème et Style ---
        # Thème sombre à fort contraste pour une meilleure accessibilité
        dark_bg = "#2b2b2b"      # Fond gris très foncé
        widget_bg = "#3c3f41"    # Fond des widgets légèrement plus clair
        text_color = "#f0f0f0"   # Texte blanc cassé, moins agressif que le blanc pur
        entry_bg = "#3c3f41"     # Fond des champs de saisie

        self.root.configure(bg=dark_bg)
        self.root.option_add("*Foreground", text_color)
        self.root.option_add("*Background", dark_bg)
        self.root.option_add("*Entry.Background", entry_bg)
        self.root.option_add("*Entry.Foreground", text_color)
        self.root.option_add("*Label.Background", dark_bg)
        self.root.option_add("*Checkbutton.Background", dark_bg)
        self.root.option_add("*Checkbutton.foreground", text_color)

        # Style pour les boutons sur le thème sombre
        self.button_style = {
            'background': '#58c1fe',   
            'foreground': 'black',       
            'activebackground': '#58a3fe', 
            'activeforeground': 'white',
            'padx': 5,
            'pady': 4
        }

        self.button_style_s = {
            'background': '#32d10a',      
            'foreground': 'black',   
            'activebackground': '#009700',
            'activeforeground': 'white',
            'padx': 5,
            'pady': 4
        }

        self.button_style_c = {
            'background': '#ffd23d',
            'foreground': 'black',
            'activebackground': '#ddd23d', 
            'activeforeground': 'black',
            'padx': 5,
            'pady': 4
        }
        # --- Fin du Thème et Style ---

        # State
        self.dossier_jeu = tk.StringVar()
        self.model_path = tk.StringVar(value=self.get_default_model_path())
        ### CHANGEMENT 1: Ajouter la variable pour le LoRA ###
        self.lora_path = tk.StringVar(value="")
        ### FIN CHANGEMENT 1 ###
        self.dossier_sortie = tk.StringVar(value="")
        self.recursif = tk.BooleanVar(value=True)
        self._worker = None  # <-- On laisse juste cette ligne
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
        
        self.log("Select the folder then click on 'Start translation'.")

    def get_default_model_path(self):
        """
        Détermine le chemin du modèle par défaut avec une logique de priorité :
        1. Le modèle par défaut spécifique s'il existe localement.
        2. Le premier autre modèle trouvé dans le dossier ./models.
        3. L'identifiant Hugging Face comme solution de repli.
        """
        specific_local_model = "./models/hub/models--virusf--nllb-renpy-rory-v6"
        models_dir = "./models"
        hf_repo_id = "virusf/nllb-renpy-rory-v6"

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
            "dossier_jeu": "",
            "model_path": self.model_path.get(),
            ### CHANGEMENT 2: Sauvegarder le LoRA ###
            "lora_path": self.lora_path.get(),
            ### FIN CHANGEMENT 2 ###
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
            ### CHANGEMENT 3: Charger le LoRA ###
            self.lora_path.set(settings.get("lora_path", ""))
            ### FIN CHANGEMENT 3 ###
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

            self.log("⚙️ Settings loaded.")
        except Exception as e:
            self.log(f"⚠️ Config file corrupted or unreadable: {e}")

    def on_closing(self):
        """Appelée lorsque l'utilisateur ferme la fenêtre."""
        self.save_settings()
        self.root.destroy()
        
    def _build_ui(self):
        # 1) Dossier des fichiers à traduire
        frame_game = tk.Frame(self.root)
        frame_game.pack(fill="x", padx=10, pady=(8, 2))
        tk.Label(frame_game, text="Folder of files to translate :").pack(side="left")
        tk.Entry(frame_game, textvariable=self.dossier_jeu, width=60).pack(side="left", padx=5)
        tk.Button(frame_game, text="Browse", command=self.choisir_dossier_jeu, **self.button_style).pack(side="left")

        # 2) Parcourir récursivement les sous-dossiers
        frame_opts = tk.Frame(self.root)
        frame_opts.pack(fill="x", padx=10, pady=(2, 2))
        tk.Checkbutton(frame_opts, text="Browse subfolders recursively", variable=self.recursif).pack(anchor="w")

        # 4) Chemin du modèle
        frame_model = tk.Frame(self.root)
        frame_model.pack(fill="x", padx=10, pady=5)
        tk.Label(frame_model, text="Model path :").pack(side="left")
        tk.Entry(frame_model, textvariable=self.model_path, width=72).pack(side="left", padx=5)
        tk.Button(frame_model, text="Browse", command=self.choisir_modele, **self.button_style).pack(side="left")


        ### CHANGEMENT 4: Ajouter le champ pour le LoRA ###
        # frame_lora = tk.Frame(self.root)
        # frame_lora.pack(fill="x", padx=10, pady=(0, 5))
        # tk.Label(frame_lora, text="LoRA path (optional):").pack(side="left")
        # tk.Entry(frame_lora, textvariable=self.lora_path, width=65).pack(side="left", padx=5)
        # tk.Button(frame_lora, text="Browse", command=self.choisir_lora, **self.button_style).pack(side="left")
        ### FIN CHANGEMENT 4 ###


        # 5) Langues (avec autodétection + ligne source/cible)
        frame_lang = tk.Frame(self.root)
        frame_lang.pack(fill="x", padx=10, pady=5)
        tk.Checkbutton(
            frame_lang,
            text="Automatically detect source language",
            variable=self.autodetect_src,
            command=self._toggle_src_field
        ).pack(anchor="w")
        line2 = tk.Frame(frame_lang)
        line2.pack(fill="x", pady=(6, 0))

        tk.Label(line2, text="Source language (NLLB code) :").pack(side="left")
        self.src_entry = tk.Entry(
            line2, textvariable=self.src_lang, width=12,
            state="disabled" if self.autodetect_src.get() else "normal"
        )
        self.src_entry.pack(side="left", padx=5)
        tk.Label(line2, text="→ Target :").pack(side="left", padx=10)
        tk.Entry(line2, textvariable=self.tgt_lang, width=12).pack(side="left", padx=5)


        # bouton aide NLLB
        tk.Button(frame_lang, text="NLLB codes", command=self._show_nllb_codes, **self.button_style).pack(anchor="w", pady=(6,0))


        # 6) Actions
        frame_actions = tk.Frame(self.root)
        frame_actions.pack(fill="x", padx=10, pady=8)
        tk.Button(frame_actions, text="Start translation", command=self.lancer_traduction, **self.button_style_s).pack(side="left")
        tk.Button(frame_actions, text="Clear log", command=self.clear_log, **self.button_style_c).pack(side="left", padx=10)

        # 7) Log
        self.logbox = scrolledtext.ScrolledText(self.root, height=22, wrap="word", font=("Consolas", 10))
        self.logbox.pack(fill="both", expand=True, padx=10, pady=10)


    # ---- Fenêtre d’aide NLLB ----
    def _show_nllb_codes(self):
        win = tk.Toplevel(self.root)
        win.title("NLLB language codes")
        win.geometry("520x640")
        # thème sombre minimal
        bg = "#2b2b2b"; fg = "#f0f0f0"; boxbg = "#3c3f41"
        for w in (win,):
            w.configure(bg=bg)

        # Barre de recherche
        frm_top = tk.Frame(win, bg=bg)
        frm_top.pack(fill="x", padx=10, pady=(10,6))
        tk.Label(frm_top, text="Search:", bg=bg, fg=fg).pack(side="left")
        q = tk.StringVar()
        ent = tk.Entry(frm_top, textvariable=q, width=32, bg=boxbg, fg=fg, insertbackground=fg)
        ent.pack(side="left", padx=8)

        # Liste + scrollbar
        frm_mid = tk.Frame(win, bg=bg)
        frm_mid.pack(fill="both", expand=True, padx=10, pady=6)
        sb = tk.Scrollbar(frm_mid)
        lb = tk.Listbox(frm_mid, height=24, activestyle="dotbox")
        sb.pack(side="right", fill="y")
        lb.pack(side="left", fill="both", expand=True)
        lb.config(bg=boxbg, fg=fg, selectbackground="#58c1fe", selectforeground="black")
        lb.config(yscrollcommand=sb.set); sb.config(command=lb.yview)

        # Remplissage initial (tri alpha sur libellé)
        items = sorted([(code, NLLB_CODES[code]) for code in NLLB_CODES], key=lambda x: x[1].lower())
        def refresh_list():
            term = (q.get() or "").strip().lower()
            lb.delete(0, "end")
            for code, name in items:
                line = f"{code}  —  {name}"
                if not term or term in code.lower() or term in name.lower():
                    lb.insert("end", line)

        refresh_list()
        ent.bind("<KeyRelease>", lambda e: refresh_list())

        # Actions bas
        frm_bot = tk.Frame(win, bg=bg)
        frm_bot.pack(fill="x", padx=10, pady=(6,10))

        def _selected_code():
            sel = lb.curselection()
            if not sel: return None
            txt = lb.get(sel[0])
            return txt.split("  —  ", 1)[0].strip()

        def to_source():
            code = _selected_code()
            if code:
                # Si l'autodetect est actif, on désactive pour pouvoir mettre la source
                if self.autodetect_src.get():
                    self.autodetect_src.set(False)
                    self._toggle_src_field()
                self.src_lang.set(code)

        def to_target():
            code = _selected_code()
            if code:
                self.tgt_lang.set(code)

        def copy_clipboard():
            code = _selected_code()
            if code:
                self.root.clipboard_clear()
                self.root.clipboard_append(code)
                try:
                    self.root.update()  # pour stabiliser le presse-papiers sous Windows
                except Exception:
                    pass

        btn_src = tk.Button(frm_bot, text="Send to Source", command=to_source,
                            bg="#58c1fe", fg="black", activebackground="#58a3fe", activeforeground="white", padx=8, pady=4)
        btn_tgt = tk.Button(frm_bot, text="Send to Target", command=to_target,
                            bg="#32d10a", fg="black", activebackground="#009700", activeforeground="white", padx=8, pady=4)
        btn_cp  = tk.Button(frm_bot, text="Copy code", command=copy_clipboard,
                            bg="#ffd23d", fg="black", activebackground="#ddd23d", activeforeground="black", padx=8, pady=4)

        btn_src.pack(side="left")
        btn_tgt.pack(side="left", padx=10)
        btn_cp.pack(side="right")

        # double-clic = envoyer vers Target (pratique)
        lb.bind("<Double-Button-1>", lambda e: to_target())
        ent.focus_set()



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


    ### CHANGEMENT 5: Ajouter la fonction pour choisir un dossier LoRA ###
    def choisir_lora(self):
        d = filedialog.askdirectory()
        if d: self.lora_path.set(d)
    ### FIN CHANGEMENT 5 ###


    def lancer_traduction(self):
        if getattr(self, '_worker', None) and self._worker.is_alive():
            messagebox.showinfo("Patiente", "Une traduction est déjà en cours.")
            return

        # Forcer le téléchargement des modèles dans ./models en utilisant la nouvelle variable
        os.environ["HF_HOME"] = os.path.abspath("./models")
        
        chemin = self.dossier_jeu.get().strip()
        modele = self.model_path.get().strip()
        ### CHANGEMENT 6: Récupérer le chemin du LoRA ###
        lora = self.lora_path.get().strip()
        ### FIN CHANGEMENT 6 ###
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
                "Indique un dossier local (ex: ./models/nllb) ou un ID HF (ex: virusf/nllb-renpy-rory-v6).")
            return

        self._stop_flag = False
        # C'EST ICI LA BONNE LIGNE À MODIFIER
        self._worker = threading.Thread(
            target=self._job_traduction, 
            args=(chemin, modele, sortie, recurse, src, tgt, lora), # <-- On ajoute 'lora' ici
            daemon=True
        )
        self._worker.start()

    def _job_traduction(self, chemin, modele, sortie, recurse, src_lang, tgt_lang, lora_path):
        ui_writer = _TextBoxWriter(self._log_queue)
        self._old_out, self._old_err = sys.stdout, sys.stderr
        sys.stdout = _TeeStream(self._old_out, ui_writer)
        sys.stderr = _TeeStream(self._old_err, ui_writer)

        try:
            print(f"📁 Folder : {chemin}")
            print(f"🧠 Model : {modele}")

            ### CHANGEMENT 9: Afficher le LoRA s'il est utilisé ###
            if lora_path:
                print(f"   + LoRA: {lora_path}")
            ### FIN CHANGEMENT 9 ###

            print(f"🌐 Languages : {src_lang} → {tgt_lang}")
            print("📂 Mode :", "Recursive" if recurse else "This folder only")
            if sortie:
                print(f"📤 Output folder : {sortie}")
            else:
                print("✍️ Overwriting files (auto backup .backup)")

            rpy_files = self._collect_rpy_files(chemin, recursive=recurse)
            if not rpy_files:
                print("ℹ️ No .rpy/.txt files found.")
                return

            print(f"🔎 Files found : {len(rpy_files)}")

            try:
                mod = importlib.import_module('traducteur_renpy_wrapper')
                TraducteurRenPy = getattr(mod, 'TraducteurRenPy')
            except Exception:
                mod = importlib.import_module('traducteur_renpy')
                TraducteurRenPy = getattr(mod, 'TraducteurRenPy')

            try:
                stop_evt = threading.Event()
                # Activer heartbeat uniquement si ce n'est PAS un repo HF (donc modèle local déjà dispo)
                if not _is_hf_repo_id(modele):
                    def _hb():
                        while not stop_evt.wait(10.0):
                            print("   ⏳ Toujours en cours de chargement du modèle…", flush=True)
                    hb_thread = threading.Thread(target=_hb, daemon=True)
                    hb_thread.start()
                else:
                    hb_thread = None  # Pas de heartbeat en cas de téléchargement HF

                try:
                    traducteur = TraducteurRenPy(
                        modele,
                        src_lang=src_lang,
                        tgt_lang=tgt_lang,
                        lora_path=lora_path if lora_path else None
                    )
                finally:
                    stop_evt.set()
                    if hb_thread:
                        try:
                            hb_thread.join(timeout=0.1)
                        except Exception:
                            pass

                # try:
                #     setattr(traducteur, "auto_install_languagetool", False)
                # except Exception:
                #     pass
                # try:
                #     setattr(traducteur, "enable_fr_grammar", bool(self.grammar_fr.get()))
                # except Exception:
                #     pass
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
                        print(f"✏️  [{idx}/{len(rpy_files)}] Écrase : {rel}")
                        print(f"✏️  (backup: {os.path.basename(backup_path)})")
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