# MedSearch AI — L'histoire complète : du problème à la solution

Ce document raconte le projet de A à Z : pourquoi il existe, quel problème il résout,
comment chaque pièce fonctionne — expliqué simplement, avec les détails techniques
quand il le faut. Compagnon du [README.md](../README.md) et du rapport de
benchmark [evaluation/RESULTS.md](../evaluation/RESULTS.md).

---

## 1. Le point de départ : un problème réel dans les hôpitaux

Imagine un médecin aux urgences, à 3h du matin. Un patient arrive avec une
combinaison bizarre de symptômes. Le médecin se dit :

> _"J'ai déjà vu un cas comme ça... il y a quelques années. Qu'est-ce qu'on avait
> fait ? Comment ça avait fini ?"_

Ce souvenir existe. Il est écrit quelque part — dans les archives de l'hôpital,
parmi des millions de comptes-rendus, de lettres de sortie, de notes cliniques.
Mais il est **introuvable**. Pourquoi ?

### Le problème n°1 : la recherche par mots-clés ne marche pas en médecine

La médecine dit la même chose de mille façons différentes :

- « crise cardiaque » = « infarctus du myocarde » = « IDM » = « STEMI » = « douleur
  thoracique constrictive avec élévation de troponine »
- « essoufflement » = « dyspnée » = « SOB » (shortness of breath)

Un moteur de recherche classique (type Ctrl+F géant) cherche les _mots exacts_.
Si le médecin tape « crise cardiaque » et que le dossier dit « infarctus », il ne
trouve **rien**. Les archives médicales sont donc des cimetières de connaissances :
on y écrit tout, on n'y retrouve rien.

### Le problème n°2 : la similarité entre patients est multidimensionnelle

Deux patients se ressemblent par leurs symptômes, leurs antécédents, leurs
médicaments, leurs résultats de labo, leur âge... Aucune requête de base de données
classique ne sait exprimer _« trouve-moi des patients dont le tableau clinique
global ressemble à celui-ci »_.

### La question du projet

> **Peut-on construire un moteur qui comprend le _sens_ d'une description clinique
> et retrouve les patients les plus similaires — avec une explication de pourquoi
> chaque cas est pertinent ?**

Une contrainte non négociable dès le départ : le système **ne diagnostique jamais**.
Il retrouve des _preuves historiques_ (des cas passés, ce qui a été fait, comment ça
a fini). La décision reste au médecin. C'est ce qu'on appelle un système d'aide à
la décision clinique (CDSS).

---

## 2. La solution en une phrase

> Transformer chaque dossier médical en une **empreinte mathématique de son sens**,
> puis comparer ces empreintes — au lieu de comparer des mots.

C'est ce qu'on appelle la **recherche sémantique**. Le reste du projet, c'est cette
idée poussée à son maximum de qualité, mesurée à chaque étape.

---

## 3. Le concept clé à comprendre : l'embedding (l'empreinte de sens)

C'est LA brique fondamentale, alors prenons le temps.

Un **modèle d'embedding** est une IA qui lit un texte et le convertit en une liste
de nombres (un « vecteur ») — ici, 1024 nombres. La magie : ces nombres encodent le
**sens** du texte, pas ses mots.

Analogie : imagine une carte géographique géante où chaque texte a une adresse GPS.
Le modèle place « crise cardiaque » et « infarctus du myocarde » **à deux adresses
voisines**, parce qu'ils veulent dire la même chose — alors qu'ils ne partagent
aucun mot. « Fracture du poignet » habite très loin, dans un autre quartier.

Chercher des patients similaires devient alors : _placer la requête du médecin sur
la carte, et regarder quels dossiers habitent à côté._

Le modèle utilisé : **BGE-M3**, un modèle open-source réputé pour la recherche.
Trois raisons de ce choix :

1. Il est **multilingue** — une requête en français retrouve des cas écrits en anglais
   (testé et vérifié : « douleur thoracique chez un homme âgé » fonctionne).
2. Il accepte des textes longs (les dossiers médicaux sont longs).
3. Il produit **deux types d'empreintes en même temps** — et c'est crucial pour la suite.

### Les deux empreintes : dense et sparse

- L'empreinte **dense** (les 1024 nombres) capture le _sens global_. Forte sur les
  synonymes et paraphrases. Faible sur les termes rares et exacts.
- L'empreinte **sparse** (une liste de mots-importants pondérés) capture les
  _termes exacts_. Si la requête contient « warfarine » ou « Takotsubo », elle
  verrouille ces mots précis.

Pourquoi les deux ? Parce qu'en médecine, il faut les deux : comprendre que
« essoufflement » ≈ « dyspnée » (dense), ET ne jamais rater le nom exact d'un
médicament ou d'une maladie rare (sparse). Chacune couvre l'angle mort de l'autre.

---

## 4. Les données : sur quoi on cherche

- **Source** : le dataset public _augmented-clinical-notes_ — 30 000 cas cliniques
  réels, dé-identifiés, issus de publications médicales. Aucune donnée personnelle.
- **Nettoyage** (script [clean_dataset.py](../data/scripts/clean_dataset.py)) :
  suppression des doublons, filtrage des textes inexploitables, extraction pour
  chaque cas de champs structurés : **sexe, âge, tranche d'âge, issue** (améliorée /
  détériorée / décédée / inconnue).
- **Résultat : 24 348 cas propres**, un fichier JSONL où chaque ligne = un cas avec
  son texte complet et ses métadonnées.
- **Calcul des empreintes** : passer 24 348 documents dans BGE-M3 prendrait des
  heures sur un CPU. Solution gratuite : un notebook **Kaggle avec GPU T4**
  ([kaggle_embed_notebook.py](../data/scripts/kaggle_embed_notebook.py)) fait le
  travail en ~30 minutes et exporte toutes les empreintes.

---

## 5. Où stocker les empreintes : la base vectorielle (Qdrant)

Comparer la requête aux 24 348 empreintes une par une serait lent. Une **base de
données vectorielle** est spécialisée exactement pour ça : elle organise les
vecteurs (avec un index appelé HNSW — pense à un réseau d'autoroutes entre
quartiers voisins de la « carte ») pour trouver les plus proches voisins en
**quelques millisecondes**.

On a choisi **Qdrant** parce qu'il sait :

1. Stocker les **deux empreintes** (dense + sparse) sur le même point.
2. Attacher à chaque point un **payload** (sexe, âge, issue...) et **filtrer pendant
   la recherche** — pas après. Quand le médecin demande « femmes de plus de 60 ans
   uniquement », la recherche ne parcourt _que_ ce sous-ensemble. Aucune fuite
   possible (vérifié par les tests).

Chargement : [load_qdrant.py](../data/scripts/load_qdrant.py) crée la collection
`cases_v1` et insère les 24 348 points. Détail propre : les IDs sont déterministes
(le même cas donne toujours le même ID), donc relancer le script ne crée jamais de
doublons.

---

## 6. La recherche, étape par étape (mode rapide)

Ce qui se passe quand le médecin tape une requête et appuie sur Entrée :

```
   "elderly man with crushing chest pain"
              │
              ▼
   ① BGE-M3 encode la requête → empreinte dense + empreinte sparse    (~1 s)
              │
              ▼
   ② Construction des filtres (sexe/âge/issue choisis par l'utilisateur)
              │
      ┌───────┴────────┐
      ▼                ▼
   ③ Recherche      Recherche
     DENSE            SPARSE          (les 2 en parallèle dans Qdrant, ~50 ms)
     top 100          top 100
      └───────┬────────┘
              ▼
   ④ FUSION : score final = 0.4 × score_dense + 0.6 × score_sparse
              │
              ▼
   ⑤ Top 10 affichés avec score, âge, sexe, issue, extrait du texte
```

L'étape ④ mérite une explication. On a deux classements (un par empreinte) —
comment les combiner ? On normalise les scores entre 0 et 1, puis on fait une
moyenne pondérée. Le poids (α = 0.4) n'a pas été deviné : on a **testé 0.3, 0.4,
0.5, 0.6, 0.7** sur notre jeu d'évaluation, et 0.4 a gagné.

Résultat mesuré de cette fusion : la recherche dense seule retrouve le bon cas dans
le top-10 **62%** du temps, la sparse seule **73%** — les deux fusionnées : **86%**.
La preuve que les deux empreintes se complètent.

---

## 7. Le mode « Thorough » : le reranker (l'arme lourde)

### Le problème restant

La recherche par empreintes a une limite fondamentale : chaque texte est résumé en
**un seul vecteur, calculé à l'avance**, sans savoir quelle question on lui posera.
C'est comme juger la ressemblance de deux livres en comparant leurs résumés de
quatrième de couverture : rapide, mais on rate des nuances.

### La solution : le cross-encoder

Un **reranker (cross-encoder)** fait l'inverse : il lit la requête ET le dossier
**ensemble, mot par mot**, et note leur pertinence l'un pour l'autre. C'est comme
donner les deux textes à un lecteur attentif et lui demander « ça correspond ? ».
Infiniment plus précis — mais trop lent pour parcourir 24 348 cas.

D'où la stratégie en deux temps, classique des moteurs de recherche sérieux :

1. La recherche rapide (section 6) **présélectionne 50 candidats** ;
2. Le reranker relit ces 50 finement et **réordonne**.

### Exemple concret observé pendant les tests

Requête : _« patient on warfarin presenting with melena »_ (méléna = sang digéré
dans les selles).

- Mode rapide : le top contenait des cas de saignement « proches » mais pas exacts.
- Mode thorough : le reranker a **remonté les cas décrivant littéralement un
  méléna** — il a lu le texte et compris la nuance que le vecteur avait floutée.

### Le détail d'ingénierie

Faire tourner un cross-encoder sur notre CPU prenait **~3 minutes par requête**
(inutilisable). Solution : l'API hébergée **NVIDIA NIM** — le modèle
`llama-nemotron-rerank-1b-v2` tourne sur leurs GPU, on l'appelle par internet,
réponse en **~0,4 seconde** pour les 50 candidats. Qualité GPU depuis une app CPU.

Trois réglages, tous choisis par mesure (jamais au feeling) :

- **Profondeur 50** : combien de candidats relire (testé de 10 à 100) ;
- **Texte tronqué à 1 600 caractères** par cas — découverte contre-intuitive :
  donner PLUS de texte (4 000 caractères) a **dégradé** la précision (le récit
  superflu noie le signal) ;
- **Mélange des scores** : 90% reranker + 10% recherche initiale (léger garde-fou).

Et si l'API NVIDIA tombe en panne ? **Fail-soft** : l'app le note dans les logs et
renvoie le classement du mode rapide. La recherche ne casse jamais.

### L'impact mesuré

|                                 | Sans reranker | Avec reranker    |
| ------------------------------- | ------------- | ---------------- |
| Le bon cas est n°1              | 48% du temps  | **93%** du temps |
| Le bon cas est dans le top 10   | 86%           | **95%**          |
| Qualité de classement (nDCG@10) | 0.66          | **0.94**         |

Coût pour l'utilisateur : **+0,5 seconde**. C'est le toggle « Thorough mode » dans
l'interface.

### Bonus : le moteur comprend « PAS de fièvre »

Les moteurs de recherche classiques ont un angle mort connu : la **négation**.
Si un médecin cherche « douleur abdominale, **pas de fièvre** », un moteur naïf
remonte quand même des cas fébriles — le mot « fièvre » est là, il matche.
Pire : un cas qui dit « le patient **nie** toute douleur thoracique » matche
parfaitement la requête « douleur thoracique ».

MedSearch intègre une couche de négation (inspirée de l'algorithme **NegEx**,
un standard du NLP clinique) qui agit après la recherche :

1. Elle repère les déclencheurs de négation dans la requête ET dans les cas
   (« no », « denies », « without », « ruled out », « no evidence of »…) et
   délimite leur portée (jusqu'à la ponctuation ou un « but »/« however ») ;
2. Elle compare : si la requête demande « pas de fièvre » et que le cas
   affirme une fièvre — **conflit** → le score du cas est pénalisé (−25% par
   conflit) et le résultat est marqué d'un badge `negation_conflict` visible
   dans l'API ;
3. Inversement, un cas qui nie la même chose que la requête reçoit un léger
   bonus (+5%).

Mesuré sur les 99 requêtes de référence, via l'API réelle : **zéro régression**
(le mode thorough garde exactement son nDCG 0.942), et le mode rapide gagne
même +2 points de Recall@5 (0.758 → 0.778) — pénaliser les cas contradictoires
fait mécaniquement remonter les bons.

---

## 8. Le bouton « Explain » : l'IA qui justifie (RAG)

Un classement sans explication n'a aucune valeur clinique : le médecin doit savoir
**pourquoi** ce cas est proposé avant de lui faire confiance.

### Le danger à éviter : l'hallucination

Les LLM (type ChatGPT) inventent avec assurance. En médecine, c'est disqualifiant.
La parade s'appelle **RAG** (Retrieval-Augmented Generation) : au lieu de demander
au LLM ce qu'il « sait », on lui **fournit les textes des cas retrouvés** et on
l'oblige à ne raisonner **que** là-dessus.

### Comment on l'oblige, concrètement (3 verrous)

**Verrou 1 — le prompt.** Les instructions envoyées au LLM (Llama-3.3-70B, via
l'API NVIDIA) imposent : n'utilise QUE les textes fournis ; CHAQUE affirmation doit
citer son cas source `[acn-12345]` ; ne diagnostique JAMAIS ; si l'info manque,
écris « preuves insuffisantes dans les cas récupérés » ; réponds en JSON structuré.

**Verrou 2 — la validation automatique.** La réponse du LLM est inspectée par le
code avant d'être montrée :

- JSON valide et conforme au schéma ? Sinon → rejet.
- Chaque citation correspond à un cas réellement fourni ? Sinon → rejet.
- Aucune formulation diagnostique interdite (« le patient a... », « vous devriez
  prescrire... ») ? Détection par expressions régulières. Sinon → rejet.
- En cas de rejet : nouvelle tentative (2× le modèle principal, puis 1× un modèle
  de secours).

**Verrou 3 — la dégradation gracieuse.** Si les 3 tentatives échouent, l'app
n'affiche **pas de prose du tout** — juste les résultats structurés (scores, cas).
Plutôt aucune explication qu'une explication risquée.

### Ce que ça donne

Pour chaque cas : les **facteurs de similarité** (cités), les **différences**
(cités), les **traitements observés dans ce cas historique et leur issue** (cités),
un niveau de confiance, et un disclaimer fixe (« ceci est une preuve historique,
pas un diagnostic »).

Moment vérifié pendant les tests : l'IA a signalé d'elle-même _« la requête
mentionne un méléna, mais le cas historique décrit une hématochézie »_ — une
distinction médicale fine, honnêtement rapportée au lieu d'être maquillée.

Dernier détail : chaque explication générée est mise en **cache Redis** — la
redemander coûte ~50 ms au lieu de ~1 minute.

---

## 9. Comment on prouve que ça marche : l'évaluation

Affirmer « ça marche bien » ne suffit pas. Il faut un protocole.

### Le jeu de test (gold set)

On a fait générer par un LLM **99 requêtes de test** : pour 99 cas tirés au hasard
(échantillonnage stratifié, graine fixe = reproductible), le LLM a écrit une
description _reformulée_ du cas, comme un médecin la taperait — **sans réutiliser
les mots du dossier**. La reformulation est cruciale : elle empêche la recherche
par mots exacts de gagner par triche.

### Le protocole (self-retrieval)

Pour chaque requête, on connaît LE cas source. On lance la recherche et on regarde
à quel rang il ressort. S'il est n°1 → parfait. Métriques standards : Recall@k
(est-il dans le top k ?), MRR et nDCG (récompensent les rangs élevés).

### L'ablation : chaque étage doit prouver son utilité

On mesure chaque configuration séparément — si un étage n'apporte rien, on le voit :

| Le bon cas dans le top 10 | Dense seul | Sparse seul | Fusion | + Reranker |
| ------------------------- | ---------- | ----------- | ------ | ---------- |
| Recall@10                 | 62%        | 73%         | 86%    | **95%**    |
| nDCG@10                   | 0.44       | 0.57        | 0.66   | **0.94**   |

Le cahier des charges initial visait Recall@10 ≥ 0.85 et nDCG ≥ 0.80. **Les deux
cibles sont dépassées.**

Tout est reproductible : `python evaluation/run_eval.py` régénère le rapport, et
[manual_test_suite.py](../evaluation/manual_test_suite.py) rejoue 14 tests sur
l'app vivante (filtres, abréviations, français, requêtes absurdes, citations...).

---

## 10. L'architecture technique (pour les curieux)

```
  Navigateur ──► React + TypeScript (port 3000, servi par nginx)
                        │  JSON/HTTP
                        ▼
                 FastAPI (port 8000)
                 ├─ /search  → BGE-M3 (en mémoire) → Qdrant (dense+sparse+filtres)
                 │             → fusion → [option: reranker NVIDIA NIM]
                 ├─ /explain → contexte → Llama-3.3-70B (NVIDIA) → validation → Redis
                 ├─ /cases/{id} → lecture directe du fichier de cas
                 └─ /health  → état de tous les composants
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
     Qdrant           Redis        APIs NVIDIA NIM
  (base vectorielle) (cache)     (reranker + LLM, cloud)
```

Le tout tourne en **Docker Compose** : `docker compose up -d` et c'est parti.

### Ce qui se passe au démarrage de l'API (dans l'ordre)

1. **Index des cas** (~1 s) : on note la position de chaque cas dans le fichier —
   servir un dossier complet = un seul saut de lecture, zéro base de données.
2. **Connexion Redis** (tolère l'échec : sans cache, l'app marche quand même).
3. **Cache des textes pour le reranker** (~1 min, en tâche de fond) : les 24 348
   textes tronqués chargés en RAM (~40 Mo). Leçon apprise : les lire depuis le
   disque à chaque requête coûtait 50 secondes — maintenant c'est 0.
4. **Chargement de BGE-M3** (~1-2 min, le plus lent) : fait UNE fois ; ensuite
   chaque requête ne coûte que ~1 s d'encodage.

### Budget latence (à chaud)

| Étape                          | Mode rapide       | Mode thorough       |
| ------------------------------ | ----------------- | ------------------- |
| Encodage de la requête         | ~1 s              | ~1 s                |
| Recherche Qdrant (×2) + fusion | ~50 ms            | ~50 ms              |
| Appel reranker NVIDIA          | —                 | ~0,4-0,5 s          |
| **Total ressenti**             | **~1-3 s**        | **rapide + ~0,5 s** |
| Explain (1re fois / en cache)  | ~30-90 s / ~50 ms | idem                |

---

## 11. La sécurité et les garde-fous (résumé)

- **Jamais de diagnostic** — verrouillé à 3 niveaux : prompt, validation par regex,
  disclaimer obligatoire.
- **Citations vérifiées par le code** — une affirmation non sourcée est rejetée.
- **Fail-soft partout** — reranker en panne → classement rapide ; Redis en panne →
  pas de cache mais app fonctionnelle ; LLM en panne → résultats sans prose.
- **Validation des entrées** — chaque requête API est contrôlée (longueurs, bornes).
- **Vie privée** — données publiques dé-identifiées ; la clé API vit dans `.env`,
  jamais dans le code.

---

## 12. Les limites (assumées)

1. **Abréviations ambiguës** : « CP » peut matcher _chronic pancreatitis_ au lieu
   de _chest pain_. Un module de NLP clinique dédié (spécifié mais hors périmètre)
   le résoudrait.
2. ~~Pas de gestion de la négation en recherche~~ → **Résolue** : une couche
   NegEx (section 7) détecte les négations dans la requête et les cas, pénalise
   les conflits et les signale par un badge. Limite résiduelle : les variantes
   morphologiques (« radiating » vs « radiation ») et les synonymes ne sont pas
   couverts — un module de NLP clinique complet le ferait.
3. **Explain multi-cas séquentiel** : 3 cas = 3 appels LLM ≈ minutes la première
   fois. OK pour un bouton à la demande ; à paralléliser à l'échelle.
4. **Évaluation par self-retrieval** : un bon proxy, mais des jugements de
   pertinence par de vrais cliniciens seraient l'étape supérieure.
5. **Modules entreprise non construits** (choix assumé) : authentification, OCR,
   codage ICD/SNOMED, journaux d'audit — spécifiés dans
   [PROJECT_SPECIFICATION.md](../PROJECT_SPECIFICATION.md), sans intérêt pour un
   corpus déjà propre et une app mono-utilisateur.

---

## 13. Carte des fichiers

```
backend/app/
  main.py                 séquence de démarrage, middleware
  core/config.py          tous les réglages + leur justification
  api/v1/routes.py        /search /explain /cases/{id} /health
  services/
    retrieval.py          encodage requête + recherche hybride + fusion
    rerank.py             reranker NVIDIA + cache textes + mélange de scores
    negation.py           détection des négations (NegEx) + pénalités
    llm.py                prompt, validation, cache Redis, dégradation
    cases.py              index cas → position fichier
frontend/src/
  App.tsx                 interface : recherche, toggle thorough, explications
data/scripts/
  clean_dataset.py        30K bruts → 24 348 cas propres
  kaggle_embed_notebook.py  calcul des empreintes sur GPU Kaggle
  load_qdrant.py          création de la collection + insertion
evaluation/
  build_gold_set.py       génération des 99 requêtes de test
  run_eval.py             l'ablation → RESULTS.md
  eval_nim_reranker.py    l'expérience du 0.942
  manual_test_suite.py    15 tests sur l'app vivante
  eval_api.py             éval des 99 requêtes via l'API réelle (non-régression)
  negation_api_test.py    4 tests ciblés de la couche négation
  RESULTS.md              le rapport de benchmark
docker-compose.yml        qdrant + redis + api + web
```
