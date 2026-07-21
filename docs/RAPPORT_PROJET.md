# MedSearch AI — Dossier de projet complet

> **Document de référence pour la rédaction du rapport final.** Il couvre tout :
> la genèse du projet, la problématique, la conception, les choix techniques et
> leurs justifications, le jeu de données, les technologies, les pipelines
> complets (données, recherche, reranking, négation, RAG), le protocole
> d'évaluation et ses résultats, la sécurité, le déploiement, les limites et les
> perspectives.
>
> Documents compagnons : [README.md](../README.md) (vitrine),
> [HOW_IT_WORKS.md](HOW_IT_WORKS.md) (vulgarisation),
> [PROJECT_SPECIFICATION.md](../PROJECT_SPECIFICATION.md) (cahier des charges),
> [evaluation/RESULTS.md](../evaluation/RESULTS.md) (rapport de benchmark brut).

---

## Table des matières

1. [Résumé exécutif](#1-résumé-exécutif)
2. [Contexte et problématique](#2-contexte-et-problématique)
3. [Objectifs, périmètre et non-objectifs](#3-objectifs-périmètre-et-non-objectifs)
4. [Fondements théoriques et état de l'art](#4-fondements-théoriques-et-état-de-lart)
5. [Conception générale et principes directeurs](#5-conception-générale-et-principes-directeurs)
6. [Le jeu de données](#6-le-jeu-de-données)
7. [Choix technologiques et justifications](#7-choix-technologiques-et-justifications)
8. [Pipeline 1 — Préparation des données (offline)](#8-pipeline-1--préparation-des-données-offline)
9. [Pipeline 2 — Recherche hybride (mode rapide)](#9-pipeline-2--recherche-hybride-mode-rapide)
10. [Pipeline 3 — Reranking neuronal (mode thorough)](#10-pipeline-3--reranking-neuronal-mode-thorough)
11. [Pipeline 4 — Couche de négation (NegEx)](#11-pipeline-4--couche-de-négation-negex)
12. [Pipeline 5 — Explication générée et ancrée (RAG)](#12-pipeline-5--explication-générée-et-ancrée-rag)
13. [L'API et l'interface utilisateur](#13-lapi-et-linterface-utilisateur)
14. [Évaluation : protocole, résultats, analyse](#14-évaluation--protocole-résultats-analyse)
15. [Sécurité, robustesse et éthique](#15-sécurité-robustesse-et-éthique)
16. [Déploiement et exploitation](#16-déploiement-et-exploitation)
17. [Limites et perspectives](#17-limites-et-perspectives)
18. [Annexes](#18-annexes)

---

## 1. Résumé exécutif

**MedSearch AI** est un moteur de recherche sémantique sur **24 348 cas cliniques
réels dé-identifiés**. Un médecin décrit un patient en langage naturel ; le
système retrouve les cas historiques les plus similaires cliniquement, les
reclasse optionnellement avec un cross-encoder hébergé sur GPU, applique une
couche de gestion de la négation clinique, et **explique pourquoi** chaque cas
correspond — avec des citations vérifiées, ancrées dans les documents récupérés.

C'est l'implémentation, à un niveau de qualité production, du **cœur IA d'un
système d'aide à la décision clinique (CDSS)**. Contrainte fondatrice : le
système **retrouve des preuves historiques, il ne diagnostique jamais**.

**Résultats clés** (jeu de test de 99 requêtes, protocole self-retrieval) :

| Métrique  | Objectif du cahier des charges | Résultat obtenu                   |
| --------- | ------------------------------ | --------------------------------- |
| Recall@10 | ≥ 0.85                         | **0.949**                         |
| nDCG@10   | ≥ 0.80                         | **0.942**                         |
| Recall@1  | —                              | **0.929**                         |
| Latence   | interactive                    | 1–3 s (rapide), +0,5 s (thorough) |

Chaque étage du pipeline (fusion hybride, reranking, négation) a été **justifié
par une mesure** (étude d'ablation), jamais par intuition.

---

## 2. Contexte et problématique

### 2.1 Le constat

Les hôpitaux produisent des volumes massifs de texte clinique non structuré :
comptes-rendus, lettres de sortie, notes d'évolution. Cette connaissance
accumulée est **écrite mais introuvable**. Un médecin confronté à une
présentation clinique inhabituelle se demande « ai-je déjà vu un cas
similaire ? » — la réponse existe dans les archives, mais aucun outil standard
ne permet de la retrouver.

### 2.2 Pourquoi la recherche classique échoue en médecine

**Problème 1 — la variabilité terminologique.** La médecine exprime le même
concept de multiples façons : « crise cardiaque » = « infarctus du myocarde » =
« IDM » = « STEMI » = « douleur thoracique constrictive avec élévation de
troponine ». Une recherche par mots-clés (lexicale) exige une correspondance
exacte : si la requête dit « crise cardiaque » et le dossier « infarctus », le
résultat est **zéro**.

**Problème 2 — la similarité multidimensionnelle.** Deux patients se
ressemblent par leurs symptômes, antécédents, traitements, biologie, âge,
évolution... Aucune requête SQL ne sait exprimer « trouve-moi des patients dont
le tableau clinique global ressemble à celui-ci ».

**Problème 3 — la négation clinique.** Les textes médicaux sont saturés de
négations (« nie toute douleur thoracique », « pas de fièvre », « absence de
signes de... »). Un moteur naïf matche le terme nié comme s'il était affirmé —
un contresens clinique.

### 2.3 La question du projet

> Peut-on construire un moteur qui comprend le **sens** d'une description
> clinique, retrouve les patients les plus similaires dans un corpus de dizaines
> de milliers de cas, gère la négation, et **justifie** chaque résultat de façon
> vérifiable — le tout avec une qualité mesurée et des temps de réponse
> interactifs, sur une infrastructure modeste (CPU) ?

---

## 3. Objectifs, périmètre et non-objectifs

### 3.1 Objectifs mesurables (fixés avant le développement)

| #   | Objectif                                                     | Cible                  | Atteint                   |
| --- | ------------------------------------------------------------ | ---------------------- | ------------------------- |
| 1   | Qualité de recherche : Recall@10                             | ≥ 0.85                 | ✅ 0.949                  |
| 2   | Qualité de classement : nDCG@10                              | ≥ 0.80                 | ✅ 0.942                  |
| 3   | Latence de recherche interactive                             | quelques sec.          | ✅ 1–3 s                  |
| 4   | Explications 100 % ancrées (zéro hallucination non détectée) | validation automatique | ✅ 3 verrous              |
| 5   | Reproductibilité complète de l'évaluation                    | 1 commande             | ✅ `run_eval.py`          |
| 6   | Déploiement en 1 commande                                    | Docker Compose         | ✅ `docker compose up -d` |

### 3.2 Périmètre fonctionnel livré

- Recherche en langage naturel (multilingue : le français fonctionne sur un
  corpus anglais — testé).
- Filtres de métadonnées (sexe, âge, issue clinique) appliqués **pendant** la
  recherche vectorielle, pas après.
- Deux modes de recherche : **rapide** (~1–3 s) et **thorough** (reranking
  cross-encoder, +0,5 s).
- Gestion de la négation (pénalisation des résultats contradictoires + badge).
- Explication « Explain this match » : justification citée, générée par LLM,
  validée par le code ; comparaison côte à côte jusqu'à 5 cas.
- Vue détaillée de chaque cas (note complète dé-identifiée + métadonnées).
- Suite d'évaluation versionnée et rejouable.

### 3.3 Non-objectifs explicites (choix assumés)

- **Pas de diagnostic** — jamais, à aucun niveau. C'est un principe de
  conception, pas une limite technique.
- **Pas d'authentification / gestion multi-utilisateurs** : app de démonstration
  mono-utilisateur ; les modules entreprise (auth, RBAC, audit logs, OCR,
  codage ICD/SNOMED) sont **spécifiés** dans le cahier des charges
  ([PROJECT_SPECIFICATION.md](../PROJECT_SPECIFICATION.md)) mais volontairement
  non construits — sans valeur ajoutée sur un corpus public déjà propre.
- **Pas de données personnelles réelles** : corpus public dé-identifié.

---

## 4. Fondements théoriques et état de l'art

### 4.1 Les embeddings (représentations vectorielles denses)

Un modèle d'embedding est un réseau de neurones (transformer) qui projette un
texte dans un espace vectoriel de grande dimension (ici **1024**), où la
**proximité géométrique encode la proximité sémantique**. « Crise cardiaque »
et « infarctus du myocarde » deviennent deux vecteurs voisins bien qu'ils ne
partagent aucun mot. La recherche de patients similaires devient une recherche
de plus proches voisins (k-NN) dans cet espace, par similarité cosinus.

**Force** : synonymie, paraphrase, multilinguisme. **Faiblesse** : les termes
rares et exacts (noms de médicaments, maladies rares) sont « floutés » dans le
vecteur.

### 4.2 Les représentations sparses (lexicales pondérées)

Une représentation sparse est un vecteur de la taille du vocabulaire, presque
entièrement nul, où chaque terme présent reçoit un poids d'importance appris
(famille SPLADE / lexical weights). C'est l'héritier moderne de TF-IDF/BM25.
**Force** : verrouillage des termes exacts (« warfarine », « Takotsubo »).
**Faiblesse** : aucune notion de synonymie.

### 4.3 La recherche hybride

La littérature en recherche d'information montre de façon constante que
**dense + sparse > chacun seul** : chaque représentation couvre l'angle mort de
l'autre. Deux stratégies de fusion existent : **RRF** (Reciprocal Rank Fusion,
basée sur les rangs) et la **fusion pondérée des scores normalisés**. Les deux
ont été implémentées et comparées (cf. §14) — la fusion pondérée a gagné.

### 4.4 Le reranking par cross-encoder

Un bi-encodeur (embedding) encode requête et document **séparément** — rapide,
mais il compresse chaque document en un vecteur unique calculé à l'avance, sans
connaître la question. Un **cross-encoder** lit requête et document
**conjointement** (attention croisée complète mot à mot) et produit un score de
pertinence — beaucoup plus précis, mais coûteux ($O(n)$ passes de transformer
par requête). L'architecture standard des moteurs sérieux est donc en deux
temps : _retrieval_ rapide qui présélectionne, _reranking_ précis qui réordonne.

### 4.5 La négation clinique (NegEx)

L'algorithme **NegEx** (Chapman et al., 2001) est le standard du TAL clinique
pour détecter les affirmations niées : il repère des déclencheurs (« no »,
« denies », « without », « ruled out »...) et leur attribue une **portée**
(fenêtre de texte jusqu'à une ponctuation ou une conjonction adversative). Le
projet en implémente une variante légère appliquée au scoring (cf. §11).

### 4.6 Le RAG (Retrieval-Augmented Generation)

Les LLM hallucinent — inacceptable en médecine. Le RAG contraint le LLM à
raisonner **uniquement** sur des documents fournis dans le prompt, avec
citation obligatoire des sources. Le projet ajoute une couche de **validation
programmatique** des sorties (cf. §12), ce qui va au-delà du RAG standard.

---

## 5. Conception générale et principes directeurs

### 5.1 Principes de conception (appliqués partout)

1. **Ablation-driven design** : chaque étage du pipeline doit prouver son
   apport par une mesure sur le jeu d'évaluation. Aucun composant « au
   feeling ». Les hyperparamètres (α de fusion, profondeur de rerank, β de
   mélange, longueur de texte) sont tous issus de balayages mesurés.
2. **Fail-soft systématique** : la panne d'un composant dégrade la qualité,
   jamais la disponibilité. Reranker en panne → classement rapide ; Redis en
   panne → pas de cache mais app fonctionnelle ; LLM en panne → résultats sans
   prose.
3. **Jamais de diagnostic** : verrouillé à trois niveaux (prompt, validation
   regex, disclaimer).
4. **Toute affirmation générée est vérifiable** : citations contrôlées par le
   code, sinon rejet.
5. **Reproductibilité** : graines fixes, jeu d'évaluation versionné, IDs
   déterministes, un script = un rapport.

### 5.2 Architecture d'ensemble

```
  Navigateur ──► React + TypeScript (port 3000, servi par nginx)
                        │  JSON/HTTP
                        ▼
                 FastAPI (port 8000, Python)
                 ├─ /search  → BGE-M3 (en mémoire) → Qdrant (dense+sparse+filtres)
                 │             → fusion α=0.4 → [option : reranker NVIDIA NIM]
                 │             → couche négation → top-k
                 ├─ /explain → contexte → Llama-3.3-70B (NVIDIA) → validation → Redis
                 ├─ /cases/{id} → lecture directe du fichier de cas (index d'offsets)
                 └─ /health  → état de tous les composants
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
     Qdrant           Redis        APIs NVIDIA NIM
  (base vectorielle) (cache)     (reranker + LLM, cloud GPU)
```

Découpage backend en **services** à responsabilité unique :
`retrieval.py` (encodage + recherche hybride + fusion), `rerank.py`
(cross-encoder + cache textes + mélange de scores), `negation.py` (NegEx +
pénalités), `llm.py` (prompt + validation + cache + dégradation), `cases.py`
(index cas → position fichier). Configuration centralisée dans
`core/config.py` (pydantic-settings, chargée depuis `.env`).

### 5.3 Décision structurante : calcul lourd délégué au cloud GPU

L'application cible tourne sur **CPU** (machine standard). Deux opérations sont
impossibles sur CPU en temps interactif : le reranking cross-encoder (~3 min
par requête mesurées localement) et la génération LLM 70B. Solution : les
appeler via l'API hébergée **NVIDIA NIM** (GPU distant, API OpenAI-compatible)
avec fail-soft si l'API est indisponible. L'encodage de la requête (BGE-M3) et
la recherche vectorielle restent locaux (~1 s et ~50 ms respectivement).

---

## 6. Le jeu de données

### 6.1 Source

**`augmented-clinical-notes`** (Hugging Face, éditeur AGBonnet) : ~30 000 cas
cliniques réels, **dé-identifiés**, issus de publications médicales (case
reports). Licence publique, aucune donnée personnelle identifiable. Chaque
entrée contient une note clinique narrative complète (anamnèse, examen,
examens complémentaires, traitement, évolution).

### 6.2 Nettoyage — script [clean_dataset.py](../data/scripts/clean_dataset.py)

| Étape | Opération                                                                                                                            | Justification                                                       |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------- |
| 1     | Suppression des doublons                                                                                                             | éviter que le même cas apparaisse plusieurs fois dans les résultats |
| 2     | Filtrage des textes inexploitables (trop courts, corrompus)                                                                          | qualité de l'index                                                  |
| 3     | Extraction de champs structurés par cas : **sexe, âge, tranche d'âge, classe d'issue** (améliorée / détériorée / décédée / inconnue) | permettre les filtres de métadonnées pendant la recherche           |
| 4     | Attribution d'un identifiant stable `acn-<n>`                                                                                        | citations et IDs déterministes                                      |

**Résultat : 24 348 cas propres** dans un fichier JSONL
(`data/processed/cases_clean.jsonl`), une ligne = un cas = texte complet +
métadonnées.

### 6.3 Schéma d'un cas (payload)

```json
{
  "case_id": "acn-12345",
  "text": "A 67-year-old man presented with ...",
  "sex": "male",
  "age": 67,
  "age_group": "60-79",
  "outcome_class": "improved"
}
```

---

## 7. Choix technologiques et justifications

| Composant               | Choix                                                                    | Alternatives considérées                                   | Justification                                                                                                                                                                                                                                                                     |
| ----------------------- | ------------------------------------------------------------------------ | ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Modèle d'embedding      | **BGE-M3** (BAAI, open source)                                           | OpenAI text-embedding, modèles biomédicaux (PubMedBERT...) | (1) produit **dense 1024-d ET sparse en une seule passe** — la recherche hybride sans double encodage ; (2) **multilingue** — requêtes en français sur corpus anglais, vérifié ; (3) contexte long (dossiers longs) ; (4) local et gratuit, pas de dépendance API pour l'encodage |
| Base vectorielle        | **Qdrant**                                                               | FAISS, Milvus, pgvector, Pinecone                          | (1) supporte **dense + sparse sur le même point** avec requête hybride native ; (2) **pré-filtrage** des métadonnées pendant la traversée HNSW (pas de post-filtrage qui vide les résultats) ; (3) open source, auto-hébergeable en Docker, API simple                            |
| Reranker (prod)         | **nvidia/llama-nemotron-rerank-1b-v2** via API NIM                       | bge-reranker-v2-m3 local (CPU)                             | local = ~3 min/requête sur CPU (inutilisable) ; l'API NIM répond en **~0,4 s pour 50 candidats**. Le bge-reranker a servi de benchmark offline sur GPU Kaggle (nDCG 0.857) ; le modèle NIM l'a battu (0.942) et est devenu le choix production                                    |
| LLM (explications)      | **meta/llama-3.3-70b-instruct** via NIM, fallback **nemotron-super-49b** | GPT-4o, modèles locaux                                     | qualité suffisante pour synthèse citée + JSON structuré, API OpenAI-compatible, fallback automatique en cas d'échec de validation                                                                                                                                                 |
| Backend                 | **FastAPI** (Python)                                                     | Flask, Django, Node                                        | écosystème ML Python natif, validation Pydantic des entrées/sorties, async, OpenAPI auto-générée                                                                                                                                                                                  |
| Frontend                | **React + TypeScript**, servi par nginx                                  | Vue, Svelte                                                | standard industriel, typage fort                                                                                                                                                                                                                                                  |
| Cache                   | **Redis**                                                                | en mémoire, memcached                                      | TTL natif, persistance optionnelle, standard ; utilisé pour cacher les explications LLM (~50 ms au lieu de ~1 min)                                                                                                                                                                |
| Orchestration           | **Docker Compose** (4 services)                                          | Kubernetes                                                 | adapté à l'échelle du projet : `docker compose up -d` et tout tourne                                                                                                                                                                                                              |
| GPU d'appoint (offline) | **Kaggle T4** (gratuit)                                                  | Colab, location cloud                                      | embedding des 24 348 cas en ~30 min au lieu d'heures sur CPU ; benchmark du reranker offline                                                                                                                                                                                      |

---

## 8. Pipeline 1 — Préparation des données (offline)

Exécuté une seule fois (ou à chaque mise à jour du corpus) :

```
 dataset brut (30K cas, Hugging Face)
        │
        ▼  clean_dataset.py  (local, ~minutes)
 24 348 cas propres + métadonnées  (cases_clean.jsonl)
        │
        ▼  kaggle_embed_notebook.py  (GPU T4 Kaggle, ~30 min)
 empreintes dense (1024-d) + sparse pour chaque cas
        │
        ▼  load_qdrant.py  (local, ~minutes)
 collection Qdrant `cases_v1` : 24 348 points
   • vecteur dense (cosinus, index HNSW)
   • vecteur sparse (index inversé)
   • payload : sex, age, age_group, outcome_class, texte
```

Détails d'ingénierie notables :

- **IDs déterministes** : le même cas produit toujours le même ID de point —
  relancer le chargement ne crée jamais de doublons (idempotence).
- **Versionnage des embeddings** : la collection porte la version `bgem3-v1` ;
  changer de modèle d'embedding = nouvelle version, comparable dans
  l'évaluation.
- Le calcul GPU est **gratuit** (quota Kaggle) — choix économique délibéré.

---

## 9. Pipeline 2 — Recherche hybride (mode rapide)

Chemin emprunté par chaque requête (endpoint `POST /api/v1/search`) :

```
   "elderly man with crushing chest pain"     + filtres éventuels
              │
              ▼
   ① BGE-M3 encode la requête → empreinte dense + sparse          (~1 s, local)
              │
              ▼
   ② Traduction des filtres utilisateur en conditions Qdrant
      (sexe / tranche d'âge / issue) — appliquées PENDANT la recherche
              │
      ┌───────┴────────┐
      ▼                ▼
   ③ recherche       recherche
      DENSE            SPARSE       (2 requêtes parallèles à Qdrant, ~50 ms)
      top 100          top 100
      └───────┬────────┘
              ▼
   ④ FUSION PONDÉRÉE : score = 0.4 × norm(dense) + 0.6 × norm(sparse)
              │
              ▼
   ⑤ COUCHE NÉGATION (pipeline 4) sur le pool des 30 premiers
              │
              ▼
   ⑥ top-k retournés (score, métadonnées, extrait, badges qualité)
```

**Justification des choix de l'étape ④ :**

- Les scores dense et sparse ne sont pas comparables (échelles différentes) →
  normalisation min-max de chaque liste avant mélange.
- Le poids **α = 0.4** (40 % dense, 60 % sparse) a été choisi par **balayage
  mesuré** de α ∈ {0.3, 0.4, 0.5, 0.6, 0.7} sur le jeu d'évaluation.
- La fusion pondérée a été comparée à **RRF** (fusion par rangs) : elle gagne
  (+0.034 nDCG@10, +0.011 Recall@10) — cf. tableau d'ablation §14.
- **Preuve de la complémentarité** : Recall@10 dense seul = 0.62, sparse seul
  = 0.73, fusion = **0.86**.

**Pré-filtrage (et non post-filtrage) :** les conditions de métadonnées sont
transmises à Qdrant qui ne parcourt que le sous-ensemble conforme. Un
post-filtrage classique (filtrer après le top-k) peut renvoyer 0 résultat et
laisse « fuir » des candidats non conformes — vérifié impossible ici par les
tests (aucune fuite de filtre observée sur la suite de tests manuelle).

---

## 10. Pipeline 3 — Reranking neuronal (mode thorough)

Activé par le toggle « Thorough mode » de l'interface (`rerank: true`).

```
 fusion hybride → top 50 candidats
        │
        ▼  appel API NVIDIA NIM (~0,4 s)
 cross-encoder llama-nemotron-rerank-1b-v2
 lit (requête + texte du cas tronqué à 1 600 caractères) pour chacun des 50
        │
        ▼
 score final = 0.9 × norm(score_rerank) + 0.1 × norm(score_retrieval)
        │
        ▼
 couche négation (pipeline 4) → top-k
```

**Hyperparamètres — tous choisis par balayage sur le gold set :**

| Paramètre                 | Valeur                         | Comment elle a été choisie                                                                                                      |
| ------------------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| Profondeur de rerank      | 50                             | balayage de 10 à 100 : 50 = meilleur compromis qualité/latence                                                                  |
| Longueur de texte par cas | 1 600 caractères (~512 tokens) | **résultat négatif documenté** : 4 000 caractères DÉGRADE la précision (nDCG 0.798 vs 0.857) — le récit superflu noie le signal |
| Mélange β                 | 0.9 rerank + 0.1 retrieval     | balayage profondeur × β ; le résidu de score retrieval sert de garde-fou                                                        |

**Ingénierie de latence :** les 24 348 textes tronqués sont chargés en RAM au
démarrage (~40 Mo, tâche de fond de ~108 s). Leçon apprise : les relire depuis
le disque à chaque requête coûtait ~50 s ; avec le cache mémoire, 0 s.

**Fail-soft :** si l'API NVIDIA échoue (panne, quota), l'application le
journalise et renvoie le classement du mode rapide — la recherche ne casse
jamais. Le drapeau `reranked` de la réponse API dit honnêtement si le
reranking a réellement eu lieu.

**Impact mesuré** (le cœur de la valeur ajoutée) : Recall@1 passe de 0.485 à
**0.929** (le bon cas est premier 93 % du temps au lieu de 48 %), nDCG@10 de
0.658 à **0.942**. Coût utilisateur : +0,5 s.

---

## 11. Pipeline 4 — Couche de négation (NegEx)

### 11.1 Le problème

« Douleur abdominale, **pas de fièvre** » remonte des cas fébriles ; un cas qui
« **nie** toute douleur thoracique » matche la requête « douleur thoracique ».
Les embeddings comme le lexical y sont aveugles.

### 11.2 La solution — service [negation.py](../backend/app/services/negation.py)

Implémentation inspirée de **NegEx** (Chapman et al.), appliquée **après** la
recherche, sur le scoring :

1. **Détection** : liste de déclencheurs triée du plus long au plus court
   (« no evidence of », « no signs of », « no history of », « negative for »,
   « absence of », « denies », « ruled out », « without », « no », « not »...),
   avec **portée** limitée à 80 caractères et coupée à la ponctuation ou aux
   conjonctions adversatives (« but », « however », « except »...). Un terme
   n'est compté nié que si **toutes** ses occurrences le sont.
2. **Analyse de la requête** : extraction des termes positifs et des termes
   niés (bigrammes de contenu + unigrammes ≥ 5 caractères, stopwords filtrés).
3. **Comparaison et ajustement** :
   - **Conflit** (la requête nie X, le cas affirme X — ou l'inverse) → score
     × (1 − 0.25) par conflit + badge **`negation_conflict`** dans la réponse
     API (transparence : l'utilisateur voit pourquoi le score a baissé) ;
   - **Concordance** (le cas nie la même chose que la requête — une « négation
     pertinente » partagée) → léger bonus × 1.05 ;
   - re-tri par score.

### 11.3 Décisions d'ingénierie

- **Pénalité, pas exclusion** : un cas contradictoire reste visible (baissé et
  badgé) — en contexte clinique, cacher de l'information est pire que la
  déclasser.
- **Non-bloquant** : pendant les ~108 s de chargement du cache de textes au
  démarrage, la couche retombe sur les extraits (300 caractères) au lieu
  d'attendre — la recherche n'est jamais bloquée.
- Paramètres exposés en configuration (`negation_enabled`, `negation_penalty`,
  `negation_bonus`, `negation_pool`).

### 11.4 Validation (double)

- **Tests ciblés** ([negation_api_test.py](../evaluation/negation_api_test.py),
  4/4) : le cas piège « no fever, no diarrhea » chute du rang 3 au rang 7 sur
  une requête contradictoire, avec badge ; un cas « denies chest pain » est
  pénalisé sur une requête « chest pain ».
- **Non-régression sur les 99 requêtes via l'API réelle**
  ([eval_api.py](../evaluation/eval_api.py)) : mode thorough **inchangé**
  (nDCG 0.942) ; mode rapide **légèrement amélioré** (Recall@5 : 0.758 → 0.778)
  — pénaliser les cas contradictoires fait mécaniquement remonter les bons.

---

## 12. Pipeline 5 — Explication générée et ancrée (RAG)

### 12.1 Motivation

Un classement sans justification n'a pas de valeur clinique : le médecin doit
savoir **pourquoi** un cas est proposé avant de lui accorder sa confiance. Mais
les LLM hallucinent — le risque à neutraliser.

### 12.2 Le flux (endpoint `POST /api/v1/explain`)

```
 cas sélectionné(s) (jusqu'à 5)
        │
        ▼
 construction du contexte : texte complet des cas récupérés (≤ 4 500 car./cas)
        │
        ▼
 LLM Llama-3.3-70B (API NVIDIA NIM) avec prompt contraint
        │
        ▼
 VALIDATION PROGRAMMATIQUE de la sortie  ──✗──► retry (2× primaire, 1× fallback)
        │ ✓                                          │ ✗ (3 échecs)
        ▼                                            ▼
 cache Redis (TTL 1 h) + affichage           aucune prose affichée
                                             (résultats structurés seuls)
```

### 12.3 Les trois verrous anti-hallucination

**Verrou 1 — le prompt.** Instructions imposées : n'utiliser QUE les textes
fournis ; CHAQUE affirmation cite son cas source `[acn-12345]` ; ne JAMAIS
diagnostiquer ; si l'information manque, écrire « preuves insuffisantes dans
les cas récupérés » ; répondre en JSON structuré.

**Verrou 2 — la validation automatique** (dans le code, avant affichage) :

- JSON valide et conforme au schéma attendu ? sinon rejet ;
- chaque citation référence un cas **réellement fourni** ? sinon rejet
  (une affirmation non sourcée est rejetée par construction) ;
- aucune formulation diagnostique interdite (« le patient a... », « vous
  devriez prescrire... ») ? détection par expressions régulières, sinon rejet ;
- en cas de rejet : nouvelle tentative (2× modèle principal, puis 1× modèle de
  secours `nemotron-super-49b`).

**Verrou 3 — la dégradation gracieuse.** Après 3 échecs, l'application
n'affiche **aucune prose** — plutôt pas d'explication qu'une explication
risquée.

### 12.4 Contenu produit

Pour chaque cas : facteurs de similarité (cités), différences (citées),
traitements observés dans ce cas historique et leur issue (cités), niveau de
confiance, disclaimer fixe (« ceci est une preuve historique, pas un
diagnostic »).

Comportement observé en test : le LLM a signalé de lui-même « la requête
mentionne un méléna, mais le cas historique décrit une hématochézie » — une
distinction clinique fine rapportée honnêtement au lieu d'être lissée.

**Performance** : première génération ~30–90 s (LLM 70B) ; regénérations
servies par le cache Redis en ~50 ms.

---

## 13. L'API et l'interface utilisateur

### 13.1 Endpoints (FastAPI, préfixe `/api/v1`)

| Endpoint      | Méthode | Rôle                                                                                                                                                            |
| ------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/search`     | POST    | recherche ; corps : `query`, `k` (≤ 25), `rerank` (bool), `filters` (sexe/âge/issue) ; réponse : résultats scorés + drapeau `reranked` + badges `quality_flags` |
| `/explain`    | POST    | explication RAG citée pour 1 à 5 cas sélectionnés                                                                                                               |
| `/cases/{id}` | GET     | dossier complet d'un cas (lecture directe par index d'offsets — un seul saut disque, zéro base de données)                                                      |
| `/health`     | GET     | état de chaque composant (Qdrant, Redis, modèle, cache reranker)                                                                                                |

Toutes les entrées sont validées par Pydantic (longueurs, bornes, énumérations).

### 13.2 Frontend (React + TypeScript, nginx)

- Barre de recherche en langage naturel + filtres (sexe, tranche d'âge, issue) ;
- toggle « Thorough mode » (reranking) ;
- cartes de résultats : score, âge, sexe, issue, extrait, badges qualité ;
- bouton « Explain this match » par résultat + sélection multiple (≤ 5) pour
  comparaison côte à côte des justifications ;
- vue détail du cas complet.

---

## 14. Évaluation : protocole, résultats, analyse

### 14.1 Construction du jeu de test (gold set)

- **99 requêtes** générées par LLM à partir d'un échantillon **stratifié** du
  corpus (proportionnel aux classes d'issue, graine fixe 42 → reproductible).
- Pour chaque cas tiré, le LLM écrit une description **reformulée**, comme un
  clinicien la taperait, **sans réutiliser les mots du dossier**. Cette
  paraphrase est cruciale : elle empêche la recherche lexicale de gagner par
  recouvrement verbatim (triche).
- Jeu versionné dans `evaluation/gold_queries.jsonl`.

### 14.2 Protocole : self-retrieval

Pour chaque requête, LE cas source est le document pertinent connu. On lance la
recherche et on mesure à quel rang il ressort. Métriques standards :
**Recall@k** (présent dans le top k ?), **MRR@10** (rang réciproque moyen),
**nDCG@10** (qualité de classement, récompense les rangs élevés).

_Limite assumée du protocole_ : le self-retrieval est une **borne inférieure**
de l'utilité réelle — un quasi-doublon clinique classé devant le cas source
compte comme un échec ici, alors que cliniquement c'est un bon résultat.

### 14.3 L'étude d'ablation (le cœur de la démonstration)

Chaque configuration est mesurée séparément — un étage qui n'apporte rien se
voit immédiatement :

| Métrique        | Dense seul | Sparse seul | Hybrid RRF | **Fusion pondérée (α=0.4)** | + rerank bge (GPU, offline) | **+ rerank NIM (production)** |
| --------------- | ---------- | ----------- | ---------- | --------------------------- | --------------------------- | ----------------------------- |
| Recall@1        | 0.303      | 0.434       | 0.455      | 0.485                       | 0.768                       | **0.929**                     |
| Recall@5        | 0.525      | 0.647       | 0.677      | 0.758                       | 0.909                       | **0.949**                     |
| Recall@10       | 0.616      | 0.727       | 0.849      | 0.859                       | 0.929                       | **0.949**                     |
| MRR@10          | 0.387      | 0.521       | 0.557      | 0.596                       | 0.833                       | **0.939**                     |
| nDCG@10         | 0.441      | 0.570       | 0.625      | 0.658                       | 0.857                       | **0.942**                     |
| Latence moyenne | 116 ms     | 25 ms       | 25 ms      | 141 ms                      | n/a (GPU)                   | ~330 ms (API)                 |

### 14.4 Les cinq enseignements

1. **L'hybride bat chaque représentation seule** : Recall@10 passe de 0.62/0.73
   à 0.86. Preuve de la complémentarité dense/sparse.
2. **La fusion pondérée bat RRF ici** (+0.034 nDCG) ; α = 0.4 issu du balayage.
3. **Le reranking résout le problème d'ordre** : les candidats trouvés mais mal
   classés sont re-scorés avec attention complète requête-document ; Recall@1
   **double presque** (0.485 → 0.929).
4. **Résultat négatif documenté** : donner PLUS de texte au reranker (4 000
   caractères) **dégrade** la précision (nDCG 0.798 vs 0.857 à 1 600) — le
   récit superflu dilue le signal. Les résultats négatifs font partie de la
   démarche scientifique du projet.
5. **Le mélange rerank/retrieval aide** : 0.9·rerank + 0.1·retrieval, choisi
   par balayage profondeur × β.

Point de référence : le plafond du pool de candidats (recall@100 de la fusion)
est 0.9596 — le reranker de production en récupère 0.9495 dès k=10 :
**l'ordonnancement est quasi optimal par rapport à ce que le retrieval peut
fournir**.

### 14.5 Non-régression de la couche négation (via l'API de production)

Mesurée par [eval_api.py](../evaluation/eval_api.py), qui traverse le **vrai**
chemin de production (FastAPI → encodage → fusion → négation → [rerank]) :

| Mode                              | R@1   | R@5       | R@10  | MRR@10 | nDCG@10   |
| --------------------------------- | ----- | --------- | ----- | ------ | --------- |
| Rapide, avant négation (baseline) | 0.485 | 0.758     | 0.859 | 0.596  | 0.658     |
| Rapide, avec négation             | 0.485 | **0.778** | 0.869 | 0.598  | 0.663     |
| Thorough, avec négation           | 0.929 | 0.949     | 0.949 | 0.939  | **0.942** |

Zéro régression ; gain de +2 points de Recall@5 en mode rapide ; le 0.942 du
mode thorough est intact (99/99 réponses effectivement rerankées — vérifié via
le drapeau `reranked`).

### 14.6 Tests fonctionnels

[manual_test_suite.py](../evaluation/manual_test_suite.py) rejoue **15 tests**
sur l'application vivante : filtres sans fuite, abréviations, requête en
français, requêtes absurdes, citations valides, négation, etc.

### 14.7 Budget latence (mesuré, à chaud)

| Étape                                      | Mode rapide       | Mode thorough       |
| ------------------------------------------ | ----------------- | ------------------- |
| Encodage de la requête (BGE-M3, CPU)       | ~1 s              | ~1 s                |
| Recherche Qdrant (dense + sparse) + fusion | ~50 ms            | ~50 ms              |
| Appel reranker NVIDIA                      | —                 | ~0,4–0,5 s          |
| Couche négation                            | ~négligeable      | ~négligeable        |
| **Total ressenti**                         | **~1–3 s**        | **rapide + ~0,5 s** |
| Explain (1re fois / en cache)              | ~30–90 s / ~50 ms | idem                |

---

## 15. Sécurité, robustesse et éthique

- **Jamais de diagnostic** — verrouillé à 3 niveaux : prompt, validation regex,
  disclaimer obligatoire sur chaque explication.
- **Citations vérifiées par le code** — toute affirmation non sourcée est
  rejetée avant affichage.
- **Fail-soft partout** — chaque dépendance externe (NIM, Redis, LLM) a un mode
  dégradé qui préserve le service.
- **Validation des entrées** — chaque requête API contrôlée par Pydantic
  (longueurs, bornes k ≤ 25, énumérations de filtres) : pas d'injection, pas
  d'abus de ressources.
- **Vie privée** — corpus public dé-identifié ; aucune donnée personnelle ; la
  clé API NVIDIA vit dans `.env` (jamais dans le code, `.env` exclu de git,
  `.env.example` fourni).
- **Transparence** — le drapeau `reranked` dit si le reranking a réellement eu
  lieu ; les badges `quality_flags` (dont `negation_conflict`) expliquent les
  pénalités de score.

---

## 16. Déploiement et exploitation

### 16.1 Docker Compose — 4 services

| Service  | Image / rôle                  | Port |
| -------- | ----------------------------- | ---- |
| `qdrant` | base vectorielle              | 6333 |
| `redis`  | cache explications            | 6379 |
| `api`    | FastAPI + BGE-M3 + services   | 8000 |
| `web`    | React buildé, servi par nginx | 3000 |

Démarrage : `docker compose up -d`. Configuration par variables
d'environnement (`.env`).

### 16.2 Séquence de démarrage de l'API (ordre mesuré)

1. **Index des cas** (~1 s) : offsets de chaque cas dans le JSONL → servir un
   dossier = un seul saut de lecture, zéro base de données.
2. **Connexion Redis** (tolère l'échec).
3. **Cache des textes du reranker** (~108 s, en tâche de fond) : 24 348 textes
   tronqués en RAM (~40 Mo).
4. **Chargement de BGE-M3** (~2,5 min, le plus lent) : une fois ; ensuite
   chaque requête ne coûte que ~1 s d'encodage.

L'endpoint `/health` expose l'état de chaque composant pendant et après le
démarrage.

---

## 17. Limites et perspectives

### 17.1 Limites assumées

1. **Abréviations ambiguës** : « CP » peut matcher _chronic pancreatitis_ au
   lieu de _chest pain_. Un module de NLP clinique dédié (spécifié, hors
   périmètre) le résoudrait.
2. ~~Négation non gérée~~ → **résolue** par la couche NegEx (§11). Limite
   résiduelle : variantes morphologiques (« radiating » vs « radiation ») et
   synonymes non couverts par l'appariement de termes.
3. **Explain multi-cas séquentiel** : 3 cas = 3 appels LLM ≈ minutes la
   première fois. Acceptable pour un bouton à la demande ; à paralléliser à
   l'échelle.
4. **Évaluation par self-retrieval** : bon proxy reproductible, mais des
   jugements de pertinence par de vrais cliniciens seraient l'étape supérieure.
5. **Modules entreprise non construits** (choix) : authentification, RBAC,
   OCR, codage ICD/SNOMED, journaux d'audit — spécifiés dans le cahier des
   charges, sans intérêt démonstratif sur un corpus public mono-utilisateur.

### 17.2 Perspectives

- Normalisation terminologique (UMLS/SNOMED) pour abréviations et synonymes ;
- jugements de pertinence humains (protocole type TREC) ;
- parallélisation des appels Explain ;
- ingestion de documents utilisateurs (OCR + dé-identification) pour un corpus
  vivant ;
- fine-tuning du reranker sur des paires cliniques annotées.

---

## 18. Annexes

### 18.1 Paramètres de configuration (extraits de `core/config.py`, tous justifiés)

| Paramètre             | Valeur                             | Origine                              |
| --------------------- | ---------------------------------- | ------------------------------------ |
| `embed_model_name`    | BAAI/bge-m3                        | choix §7                             |
| `fusion_alpha`        | 0.4                                | balayage {0.3–0.7} sur gold set      |
| `prefetch_limit`      | 100                                | plafond de rappel du pool : 0.9596   |
| `rerank_model`        | nvidia/llama-nemotron-rerank-1b-v2 | benchmark vs bge-reranker            |
| `rerank_depth`        | 50                                 | balayage 10–100                      |
| `rerank_beta`         | 0.9                                | balayage profondeur × β              |
| `rerank_doc_chars`    | 1600                               | résultat négatif : 4000 dégrade      |
| `negation_penalty`    | 0.25                               | pénalité par conflit, multiplicative |
| `negation_bonus`      | 0.05                               | négation pertinente partagée         |
| `negation_pool`       | 30                                 | pool ajusté en mode rapide           |
| `llm_model_primary`   | meta/llama-3.3-70b-instruct        | qualité/coût                         |
| `llm_temperature`     | 0.2                                | factualité > créativité              |
| `cache_ttl_seconds`   | 3600                               | fraîcheur vs coût LLM                |
| `default_k` / `max_k` | 10 / 25                            | UX / protection ressources           |

### 18.2 Carte des fichiers

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
  eval_api.py             éval des 99 requêtes via l'API réelle (non-régression)
  manual_test_suite.py    15 tests sur l'app vivante
  negation_api_test.py    4 tests ciblés de la couche négation
  RESULTS.md              le rapport de benchmark
docker-compose.yml        qdrant + redis + api + web
PROJECT_SPECIFICATION.md  cahier des charges complet
```

### 18.3 Glossaire

| Terme                        | Définition courte                                                     |
| ---------------------------- | --------------------------------------------------------------------- |
| **Embedding**                | vecteur de nombres encodant le sens d'un texte                        |
| **Dense / Sparse**           | empreinte sémantique globale / poids de termes exacts                 |
| **Recherche hybride**        | fusion des classements dense et sparse                                |
| **HNSW**                     | index de graphe pour recherche rapide de plus proches voisins         |
| **Cross-encoder / reranker** | modèle lisant requête + document ensemble pour un score précis        |
| **NegEx**                    | algorithme de détection des négations en texte clinique               |
| **RAG**                      | génération contrainte aux documents récupérés, avec citations         |
| **Recall@k**                 | proportion de requêtes dont le bon document est dans le top k         |
| **MRR**                      | moyenne des inverses du rang du bon document                          |
| **nDCG**                     | qualité de classement, récompense les bons documents haut placés      |
| **Self-retrieval**           | protocole : la requête paraphrase un cas connu, on mesure son rang    |
| **Ablation**                 | mesurer chaque composant séparément pour prouver son apport           |
| **Fail-soft**                | une panne dégrade la qualité, jamais la disponibilité                 |
| **CDSS**                     | Clinical Decision Support System (aide à la décision, pas diagnostic) |
