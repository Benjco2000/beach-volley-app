# 🏐 beach-volley-app — analyse vidéo de la manchette

Upload d'une vidéo → **verdict (réussie / à corriger)** + **axes d'amélioration**, à partir du squelette du joueur et du ballon.

## Pipeline de bout en bout

```
Google Drive ──ingest──▶ data/raw (Bronze)        vidéos .MOV + manifeste (label, joueuse)
                │
                ▼ pose (YOLO11-pose + détection ballon)
            data/interim/*.npz (Silver)           17 points du corps + ballon, par frame
                │
                ▼ features (src/volleyai/features/manchette.py)
            data/processed/features_*.csv (Gold)  ~17 mesures biomécaniques par manchette
                │
      ┌─────────┼───────────────────┐
      ▼         ▼                   ▼
   Règles    Profil de référence   Modèle (régression logistique)
   du coach  (réussies vs ratées)  + validation croisée, MLflow
      └─────────┼───────────────────┘
                ▼
        Rapport : verdict, critères ✅/❌/➖, axes d'amélioration,
        vidéo annotée, (option) reformulation par Claude
                ▼
        App Streamlit (upload) / CLI / batch G3
```

### Les trois "cerveaux" et pourquoi
| Composant | Rôle | Pourquoi |
|---|---|---|
| **Règles** (`configs/manchette.yaml`) | Tes critères de coach traduits en seuils mesurables | Marche dès aujourd'hui, sans données, et chaque verdict est explicable |
| **Profil de référence** (`volleyai train`) | Pour chaque mesure : plage des manchettes réussies, pouvoir discriminant (AUC), seuil suggéré | Répond à « quels sont les bons paramètres d'une manchette réussie ? » et sert à recaler les seuils des règles |
| **Modèle** (`volleyai train`) | Probabilité réussie/ratée à partir des mesures | Prend le relais des règles quand il y a assez de données ; simple et régularisé car peu de vidéos |

Le LLM (optionnel) ne fait que **reformuler** le diagnostic chiffré en conseil de coach. Il ne voit pas la vidéo et ne décide pas du verdict.

### Critères mesurés (vue de profil)
| Critère coach | Mesure | Remarque |
|---|---|---|
| Contact sur les avant-bras, pas les coudes | position du ballon sur le segment poignet→coude | nécessite que le ballon soit détecté |
| Pas de swing des bras | montée de l'angle bras/tronc autour du contact | |
| Jambes et bras synchronisés | décalage entre pic de poussée des hanches et pic de montée des bras | |
| Ne pas se redresser | remontée des épaules autour du contact | |
| Ne pas sauter | décollement simultané des deux chevilles | |
| Placement des jambes | écart avant/arrière des pieds | ⚠ le « ballon à droite → jambe droite derrière » demande une vue de face (voir plus bas) |
| (+) Bras tendus, jambes fléchies | angles coude / genou | |

Chaque vidéo peut contenir **plusieurs manchettes** : elles sont détectées et analysées une par une.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[app,drive,dev]"     # + ".[llm]" pour Claude, ".[mlflow]" pour MLflow
pytest                                # tests sur des manchettes synthétiques
```
Les poids YOLO (~6 Mo) se téléchargent automatiquement au premier lancement.

### Accès Google Drive (compte de service)
1. Google Cloud Console → nouveau projet → activer **Google Drive API**.
2. IAM → Comptes de service → créer → Clés → **JSON** → enregistrer sous `secrets/service_account.json` (ignoré par Git).
3. Dans Drive, partager le dossier **BEACH VOLLEY** en *Lecteur* avec l'e-mail du compte de service.

Sans API : synchronise le dossier avec Google Drive pour ordinateur et utilise `--local-dir`.

## Utilisation

```bash
volleyai ingest --split train         # VIDEO MANCHETTES/{réussie,ratée} -> data/raw/train
volleyai extract --split train        # pose + features -> data/processed/features_train.csv
volleyai train                        # modèle + reports/reference_manchette.json
volleyai ingest --split test          # Vidéo G3/<joueuse>/*Manchettes*
volleyai evaluate-g3                  # -> reports/g3_manchettes.csv (verdict par manchette et joueuse)
volleyai analyze ma_video.MOV --overlay annotee.mp4
streamlit run app/streamlit_app.py    # l'app d'upload
```

## Ce qu'il faut ajouter pour que ça devienne fiable

1. **Plus de manchettes ratées, et pas que les tiennes.** Aujourd'hui il y a ~20 réussies (dont 7 clips nommés `IMG_6347.mov`) pour 6 ratées, toutes faites par toi. Le modèle apprendrait « ton » geste. Objectif : 50+ par classe, avec plusieurs joueurs.
2. **Labelliser les défauts, pas seulement réussie/ratée.** Remplis `annotations/labels_template.csv` (une colonne par critère). C'est ce qui permet de vérifier chaque règle séparément et, plus tard, d'apprendre un modèle par défaut.
3. **Labelliser un échantillon de G3** pour mesurer la vraie précision de l'app. Sans ça, on ne sait pas si elle a raison.
4. **Protocole de tournage** : trépied, de profil, corps entier **et ballon** dans le champ, un seul joueur au premier plan, 60 fps.
5. **Le critère gauche/droite des jambes** demande de savoir de quel côté arrive le ballon. En vue de profil c'est invisible : soit une 2e caméra (de face ou de dos), soit un tag manuel « ballon à gauche/droite » dans l'annotation.
6. **Détection du ballon** : le détecteur générique (COCO « sports ball ») peut rater un ballon de beach sur le sable. Si c'est le cas sur tes vidéos, on l'entraîne spécifiquement avec quelques centaines d'images annotées.
7. **RGPD** : ce sont des vidéos de personnes identifiables. Il faut l'accord des joueuses et les vidéos restent dans ton Drive privé (jamais dans Git, cf. `.gitignore`).

## Feuille de route
- [x] v0 : pipeline complète, règles du coach, profil de référence, modèle simple, app, batch G3
- [ ] Recaler les seuils sur tes vraies vidéos (`reports/reference_manchette.json`)
- [ ] Annotation par critère + évaluation sur G3
- [ ] Détecteur de ballon spécifique beach
- [ ] Modèle séquentiel (sur les séries de keypoints) quand il y aura >200 clips
- [ ] Autres gestes : passe, attaque (même pipeline, nouveaux critères)
- [ ] Déploiement (Databricks : Volumes + MLflow + Databricks Apps, ou Streamlit Cloud)

## Structure
```
configs/manchette.yaml          seuils, IDs des dossiers Drive, paramètres
src/volleyai/ingest/drive.py    ingestion Drive (compte de service), manifeste
src/volleyai/pose.py            YOLO11-pose + ballon, suivi du joueur principal
src/volleyai/keypoints.py       interpolation, lissage, orientation profil
src/volleyai/features/          features biomécaniques + détection des contacts
src/volleyai/analysis/          règles, profil de référence, feedback (texte / Claude)
src/volleyai/model/train.py     classifieur, validation croisée groupée, MLflow
src/volleyai/pipeline.py        orchestration + cache
src/volleyai/overlay.py         vidéo annotée (H.264)
app/streamlit_app.py            application d'upload
tests/                          générateur de manchettes synthétiques + tests
```
