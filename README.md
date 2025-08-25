# 🎮 Ren'Py Translator (NLLB)

Un outil pour **traduire automatiquement les fichiers `.txt` des jeux Ren'Py**, basé sur [NLLB (No Language Left Behind)](https://huggingface.co/facebook/nllb-200-distilled-600M) et Hugging Face Transformers.  

⚠️ Ce traducteur est conçu pour fonctionner sur les fichiers générés par l’application [Rory-RenExtract](https://github.com/Rory-Mercury-91/rory_tool).  
La compatibilité complète sera disponible dès la prochaine version de cet outil.

---

## 📦 Installation

### Prérequis
- Windows 10 - 11
- Carte Nvidia >= RTX 3060
- Min 6 Go Vram

### Installation rapide (Windows)
lance simplement : Install Requirements.bat

Cela installera automatiquement toutes les dépendances nécessaires (`torch`, `transformers`, etc.).

## 🚀 Utilisation

### Interface graphique (Windows)
lance simplement : Lancement.bat

1. **Folder of files to translate** → choisir le dossier du jeu Ren’Py.  
2. **Browse subfolders recursively** → cocher si tu veux inclure aussi les sous-dossiers.  
3. **Model path** → chemin du modèle (dossier local `./models/...` ou identifiant Hugging Face `virusf/nllb-renpy-rory-v3`).  
4. **Languages** → configurer :  
   - `Source language (NLLB code)` (ou laisser l’option **Automatically detect source language** activée).  
   - `Target` (`fra_Latn` par défaut pour français).  
5. Cliquer sur **Start translation** pour lancer.  


---

## ✨ Fonctionnalités
- Interface graphique simple (**Tkinter**).
- Traduction **en place** ou vers un **dossier de sortie**.
- **Backup automatique** des fichiers originaux (`.backup`).
- Support **récursif** des sous-dossiers.
- Détection automatique de la **langue source** (optionnelle).
- Sélection du modèle :
  - dossier local (`./script/models/...`)  
  - ou repo Hugging Face (`virusf/nllb-renpy-rory-v3` par défaut).
- Gestion VRAM optimisée (FP16, TF32, batch adaptatif).
- Protection des **tokens Ren'Py** (`RENPY_CODE_...`, `{color=}`, `{size=}`, etc.).
- Sauvegarde automatique des préférences dans `config.json`.

---

## 📁 Structure du projet
- `interface_renpy_translator.pyw` → GUI (Tkinter).
- `traducteur_renpy.py` → Cœur du traducteur optimisé.
- `traducteur_renpy_wrapper.py` → Protection des tokens/balises et post-traitements.
- `traducteur_renpy_jeu_complet.py` → Traduction batch d’un dossier entier.
- `Install Requirements.bat` → Script d’installation rapide des dépendances (Windows).
- `Lancement.bat` → Lance directement l’interface graphique sous Windows.

---

## 🔧 Paramètres utiles
- `auto` : détection auto de la langue source.
- `fra_Latn` : français (cible par défaut).
- `eng_Latn`, `jpn_Jpan`, `zho_Hans`, etc. : autres codes de langues NLLB.  
👉 [Liste complète des langues supportées](https://huggingface.co/facebook/nllb-200-distilled-600M).

---

## ⚠️ Limitations
- Nécessite un GPU **>= 8 Go VRAM** pour de bonnes performances.  
- Les traductions automatiques peuvent nécessiter une **relecture manuelle**.  

---

## 📜 Licence
Ce projet est distribué sous licence

**Creative Commons Attribution - NonCommercial 4.0 International (CC BY-NC 4.0)**.  


👉 Vous êtes libres de l’utiliser, le modifier et le partager, tant que :  
- vous citez l’auteur original,  
- vous n’en faites pas un usage commercial (vente, monétisation, etc.).  

📖 Texte complet de la licence : [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.fr)

