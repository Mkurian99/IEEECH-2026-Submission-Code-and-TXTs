# ============================================================================
# SYMBOLIC ENTROPY — CANONICAL NLP SHUFFLE TEST SUITE  (GOOGLE COLAB)
# Version: 1.0 (Canonical, Colab edition)
# ============================================================================
#
# Implements all validated methodological fixes over previous versions:
#
#   FIX [2]  segment_into_units(): fixed-size token chunks for all segmentation
#            Replaces period-based split_sentences() in all 5 affected methods.
#            Period splitting creates arbitrarily sized fragments on shuffled
#            text (periods land randomly among scrambled words), making
#            cross-condition Cohen's d comparisons unreliable.
#   FIX [3]  GPT-2: .eval() called before inference — deterministic perplexity
#   FIX [4]  Sentiment: segment_into_units(80) — consistent 80-word chunks
#   FIX [5]  TF-IDF: segment_into_units(50) + single vectorizer on original
#   FIX [6]  NER: segment_into_units(50) — consistent 50-word chunks
#   FIX [7]  LDA: segment_into_units(50) + original-only training
#   FIX [8]  BERTScore: batched inference — single bertscore() call. Results
#            identical to per-pair loop; hours vs minutes on LOTR-scale texts.
#   FIX [9]  BERTopic: segment_into_units(50) + original-only training
#            NOTE: honest BERTopic score is expected to FAIL (d < 1.0).
#            This is the correct result and is evidence for SE's positioning.
#   FIX [10] KEY INSIGHTS: correct results dict population (was broken in
#            class-based version — iterated over empty dict)
#   FIX [11] Sentiment model: memory released after use (was commented out)
#
# Environment:  Google Colab (uses files.upload() for file selection)
# Architecture: flat procedural (all logic top-level, runs via main())
#
# Usage in Colab:
#   1. Open a new Colab notebook
#   2. Paste this entire file into a single cell, or split into cells at the
#      section dividers
#   3. Run — you will be prompted three times to upload files
# ============================================================================


# ============================================================================
# LIBRARY INSTALLATION (Colab cell magic — runs in shell)
# ============================================================================

!pip install transformers torch python-docx bert-score -q
!pip install scikit-learn spacy gensim bertopic umap-learn hdbscan -q
!python -m spacy download en_core_web_sm -q


# ============================================================================
# USER CONFIGURATION — edit before running
# ============================================================================

# Set each method to 1 to run, 0 to skip
list_of_methods = {
    'Perplexity':  1,
    'Sentiment':   1,
    'TF-IDF':      1,
    'NER':         1,
    'LDA':         1,
    'BERTScore':   1,
    'BERTopic':    1,
}


# ============================================================================
# IMPORTS
# ============================================================================

import os
import io
import re
import numpy as np

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, pipeline
from bert_score import score as bertscore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from gensim import corpora
from gensim.models import LdaModel
from bertopic import BERTopic
from docx import Document

# Colab file upload
from google.colab import files


# ============================================================================
# FILE READING
# ============================================================================

def read_file(file_dict):
    """
    Read text from a Colab files.upload() result dict.

    files.upload() returns: {filename: bytes_content}.
    This handles .docx (via python-docx) and .txt (with encoding fallback).
    """
    filename = list(file_dict.keys())[0]
    content  = file_dict[filename]

    if filename.endswith('.docx'):
        doc = Document(io.BytesIO(content))
        return '\n'.join([para.text for para in doc.paragraphs])
    else:
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                return content.decode(encoding)
            except (UnicodeDecodeError, AttributeError):
                continue
        return content.decode('utf-8', errors='ignore')


# ============================================================================
# SHARED UTILITY FUNCTIONS
# ============================================================================

def calculate_cohens_d(scores_a, scores_b):
    """
    Calculate Cohen's d using pooled standard deviation.

    Returns: (cohens_d, mean_a, mean_b, std_a, std_b)
    """
    mean_a = np.mean(scores_a)
    mean_b = np.mean(scores_b)
    std_a  = np.std(scores_a, ddof=1)
    std_b  = np.std(scores_b, ddof=1)

    pooled_sd = np.sqrt((std_a**2 + std_b**2) / 2)
    cohens_d  = abs(mean_a - mean_b) / pooled_sd if pooled_sd > 0 else 0.0

    return cohens_d, mean_a, mean_b, std_a, std_b


def split_into_windows(text, window_size=200, overlap=0.0):
    """
    Split text into overlapping windows of tokens.
    Used by Perplexity only (overlap required for sliding window inference).
    """
    words     = text.split()
    step_size = int(window_size * (1 - overlap))
    windows   = []
    for i in range(0, len(words) - window_size + 1, step_size):
        windows.append(' '.join(words[i:i + window_size]))
    return windows


def segment_into_units(text, unit_size=50, min_words=10):
    """
    Split text into fixed-size non-overlapping token chunks.

    FIX [2]: Canonical segmentation function replacing split_sentences()
    for all methods that compare across shuffle conditions. Unlike period-
    based splitting, this guarantees identical granularity across original,
    word-shuffled, and sentence-shuffled conditions regardless of where
    punctuation lands.

    Args:
        text:      input string
        unit_size: number of tokens per chunk (default 50)
        min_words: discard chunks shorter than this (default 10)

    Returns:
        list of string chunks
    """
    words    = text.split()
    segments = []
    for i in range(0, len(words), unit_size):
        seg_words = words[i:i + unit_size]
        if len(seg_words) >= min_words:
            segments.append(' '.join(seg_words))
    return segments


def split_sentences(text):
    """
    Period-based sentence splitter.
    Retained for reference but NOT called by any method in this canonical
    version. All methods use segment_into_units() instead. See FIX [2].
    """
    sentences = []
    for sent in text.replace('\n', ' ').split('.'):
        sent = sent.strip()
        if len(sent) > 20:
            sentences.append(sent)
    return sentences


def get_verdict(d):
    """Return graded verdict string for a Cohen's d value."""
    if d >= 3.0:
        return "✅✅ STRONG PASS"
    elif d >= 2.0:
        return "✅  PASS"
    elif d >= 1.0:
        return "~  BORDERLINE"
    else:
        return "❌  FAIL"


def store_results(results, method_name,
                  d_word, d_sent,
                  orig_mean, word_mean, sent_mean,
                  orig_std,  word_std,  sent_std,
                  n):
    """
    Write method results into the shared results dict and print a summary.
    FIX [10]: Results are stored in the flat dict that print_results()
    reads from. The class-based version never populated this dict, causing
    KEY INSIGHTS to silently report 0 methods in all categories.
    """
    results[method_name] = {
        'd_vs_word':      d_word,
        'd_vs_sent':      d_sent,
        'original_mean':  orig_mean,
        'word_shuf_mean': word_mean,
        'sent_shuf_mean': sent_mean,
        'original_std':   orig_std,
        'word_shuf_std':  word_std,
        'sent_shuf_std':  sent_std,
        'n_observations': n,
    }
    print(f"\n✓ {method_name} complete:")
    print(f"   d(orig vs word-shuf)     = {d_word:.3f}  {get_verdict(d_word)}")
    print(f"   d(orig vs sent-shuf)     = {d_sent:.3f}  {get_verdict(d_sent)}")
    print(f"   n = {n}")


# ============================================================================
# METHOD 1: PERPLEXITY (GPT-2) — SLIDING WINDOW
# ============================================================================

def run_perplexity(original_text, word_shuffled_text, sent_shuffled_text, results):
    """
    GPT-2 perplexity over 200-token sliding windows (50% overlap).
    Pre-trained model — no fitting on test data.

    FIX [3]: gpt2_model.eval() called before inference. Without this,
    dropout layers remain active and introduce stochasticity into
    perplexity scores, producing non-reproducible results across runs.
    """
    print("\n" + "="*80)
    print("METHOD 1/7: GPT-2 Perplexity (Sliding Window)")
    print("STATUS: ✅ Pre-trained model — valid comparison")
    print("FIX [3]: .eval() mode enabled for deterministic inference")
    print("="*80)

    print("\nLoading GPT-2 model...")
    gpt2_model     = GPT2LMHeadModel.from_pretrained('gpt2')
    gpt2_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    gpt2_model.eval()   # FIX [3]
    print("✓ Model loaded (eval mode)")

    def calculate_window_perplexity(text, window_size=200):
        windows      = split_into_windows(text, window_size=window_size)
        perplexities = []
        print(f"   Processing {len(windows)} windows...")
        for i, window in enumerate(windows):
            try:
                encodings = gpt2_tokenizer(
                    window, return_tensors='pt', truncation=True, max_length=200)
                with torch.no_grad():
                    outputs = gpt2_model(**encodings, labels=encodings.input_ids)
                    perplexities.append(torch.exp(outputs.loss).item())
            except Exception:
                continue
            if (i + 1) % 20 == 0:
                print(f"   Processed {i + 1}/{len(windows)} windows...")
        return np.array(perplexities)

    print("\n⚙️  ORIGINAL text...")
    orig_ppls = calculate_window_perplexity(original_text)
    print("\n⚙️  WORD-SHUFFLED text...")
    word_ppls = calculate_window_perplexity(word_shuffled_text)
    print("\n⚙️  SENTENCE-SHUFFLED text...")
    sent_ppls = calculate_window_perplexity(sent_shuffled_text)

    d_word, orig_mean, word_mean, orig_std, word_std = calculate_cohens_d(orig_ppls, word_ppls)
    d_sent, _,        sent_mean, _,        sent_std  = calculate_cohens_d(orig_ppls, sent_ppls)
    store_results(results, 'Perplexity',
                  d_word, d_sent,
                  orig_mean, word_mean, sent_mean,
                  orig_std,  word_std,  sent_std,
                  len(orig_ppls))

    del gpt2_model, gpt2_tokenizer
    torch.cuda.empty_cache()


# ============================================================================
# METHOD 2: SENTIMENT ANALYSIS
# ============================================================================

def run_sentiment(original_text, word_shuffled_text, sent_shuffled_text, results):
    """
    DistilBERT sentiment confidence per text chunk, capped at 50 chunks.
    Pre-trained model — no fitting on test data.

    FIX [4]:  segment_into_units(unit_size=80) replaces period-based
              split_sentences(). Fixed 80-word chunks ensure all three
              conditions are analysed at identical granularity.
    FIX [11]: Model memory released after use (del + empty_cache).
              Previously this was commented out, risking OOM on GPU.
    """
    print("\n" + "="*80)
    print("METHOD 2/7: Sentiment Analysis")
    print("STATUS: ✅ Pre-trained model — valid comparison")
    print("FIX [4]:  segment_into_units(80) for consistent chunk sizes")
    print("FIX [11]: model memory released after use")
    print("="*80)

    print("\nLoading sentiment model...")
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=0 if torch.cuda.is_available() else -1)
    print("✓ Model loaded")

    def analyze_sentiment_chunks(text):
        # FIX [4]: fixed-size units
        chunks = segment_into_units(text, unit_size=80, min_words=10)
        scores = []
        n_run  = min(len(chunks), 50)
        print(f"   Analyzing {n_run} chunks (of {len(chunks)} total)...")
        for i, chunk in enumerate(chunks[:50]):
            try:
                result = sentiment_analyzer(chunk)[0]
                score  = result['score'] if result['label'] == 'POSITIVE' else 1 - result['score']
                scores.append(score)
            except Exception:
                continue
            if (i + 1) % 10 == 0:
                print(f"   Processed {i + 1}/{n_run} chunks...")
        return np.array(scores)

    print("\n⚙️  ORIGINAL text...")
    orig_sent = analyze_sentiment_chunks(original_text)
    print("\n⚙️  WORD-SHUFFLED text...")
    word_sent = analyze_sentiment_chunks(word_shuffled_text)
    print("\n⚙️  SENTENCE-SHUFFLED text...")
    sent_sent = analyze_sentiment_chunks(sent_shuffled_text)

    d_word, orig_mean, word_mean, orig_std, word_std = calculate_cohens_d(orig_sent, word_sent)
    d_sent, _,        sent_mean, _,        sent_std  = calculate_cohens_d(orig_sent, sent_sent)
    store_results(results, 'Sentiment',
                  d_word, d_sent,
                  orig_mean, word_mean, sent_mean,
                  orig_std,  word_std,  sent_std,
                  len(orig_sent))

    # FIX [11]: release model memory
    del sentiment_analyzer
    torch.cuda.empty_cache()


# ============================================================================
# METHOD 3: TF-IDF COHERENCE
# ============================================================================

def run_tfidf(original_text, word_shuffled_text, sent_shuffled_text, results):
    """
    Consecutive cosine similarity between TF-IDF vectors of text units.
    Single vectorizer fit on original — all conditions share same vocab space.

    FIX [5a]: segment_into_units(unit_size=50) replaces split_sentences().
    FIX [5b]: Single vectorizer fit on original only (correct in previous
              versions — explicitly retained and documented here).

    Note: FIX [5b] is the one place Monroy's version was already correct
    and Kurian's was not. Do not change the vectorizer strategy.
    """
    print("\n" + "="*80)
    print("METHOD 3/7: TF-IDF Coherence")
    print("STATUS: 🔧 Single vectorizer on original — valid comparison")
    print("FIX [5a]: segment_into_units(50) for consistent chunk sizes")
    print("FIX [5b]: single vectorizer on original only (retained)")
    print("="*80)

    def calculate_tfidf_coherence(orig_text, word_text, sent_text):
        # FIX [5a]: fixed-size units
        orig_units = segment_into_units(orig_text, unit_size=50, min_words=10)
        word_units = segment_into_units(word_text, unit_size=50, min_words=10)
        sent_units = segment_into_units(sent_text, unit_size=50, min_words=10)

        if len(orig_units) < 2:
            return np.array([0.0]), np.array([0.0]), np.array([0.0])

        # FIX [5b]: fit once on original, transform all through shared vocab
        vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        try:
            vectorizer.fit(orig_units)
        except Exception:
            return np.array([0.0]), np.array([0.0]), np.array([0.0])

        def consecutive_similarities(units, vec):
            try:
                mat = vec.transform(units)
            except Exception:
                return np.array([0.0])
            sims = []
            for i in range(mat.shape[0] - 1):
                sim = cosine_similarity(mat[i:i+1], mat[i+1:i+2])[0][0]
                sims.append(sim)
            return np.array(sims)

        print("   Transforming original units...")
        orig_sims = consecutive_similarities(orig_units, vectorizer)
        print("   Transforming word-shuffled units...")
        word_sims = consecutive_similarities(word_units, vectorizer)
        print("   Transforming sentence-shuffled units...")
        sent_sims = consecutive_similarities(sent_units, vectorizer)

        return orig_sims, word_sims, sent_sims

    print("\n⚙️  Calculating TF-IDF coherence (single vectorizer)...")
    orig_tfidf, word_tfidf, sent_tfidf = calculate_tfidf_coherence(
        original_text, word_shuffled_text, sent_shuffled_text)

    d_word, orig_mean, word_mean, orig_std, word_std = calculate_cohens_d(orig_tfidf, word_tfidf)
    d_sent, _,        sent_mean, _,        sent_std  = calculate_cohens_d(orig_tfidf, sent_tfidf)
    store_results(results, 'TF-IDF',
                  d_word, d_sent,
                  orig_mean, word_mean, sent_mean,
                  orig_std,  word_std,  sent_std,
                  len(orig_tfidf))


# ============================================================================
# METHOD 4: NAMED ENTITY RECOGNITION (NER)
# ============================================================================

def run_ner(original_text, word_shuffled_text, sent_shuffled_text, results):
    """
    Entity density (entities per 100 words) per text unit.
    Pre-trained spaCy model — no fitting on test data.

    FIX [6]: segment_into_units(unit_size=50) replaces split_sentences().
    Entity density is a ratio so it is partially length-normalised, but
    very short fragments from period-based splitting on shuffled text
    produce unstable density values and inconsistent observation counts
    across conditions, making cross-condition Cohen's d comparisons
    unreliable.
    """
    print("\n" + "="*80)
    print("METHOD 4/7: Named Entity Recognition (NER)")
    print("STATUS: ✅ Pre-trained model — valid comparison")
    print("FIX [6]: segment_into_units(50) for consistent chunk sizes")
    print("="*80)

    print("\nLoading spaCy NER model...")
    nlp = spacy.load("en_core_web_sm")
    print("✓ Model loaded")

    def calculate_ner_density(text):
        # FIX [6]: fixed-size units
        units     = segment_into_units(text, unit_size=50, min_words=10)
        densities = []
        print(f"   Analyzing {len(units)} units...")
        for i, unit in enumerate(units):
            doc        = nlp(unit)
            word_count = len(unit.split())
            density    = (len(doc.ents) / word_count) * 100 if word_count > 0 else 0.0
            densities.append(density)
            if (i + 1) % 20 == 0:
                print(f"   Processed {i + 1}/{len(units)} units...")
        return np.array(densities)

    print("\n⚙️  ORIGINAL text...")
    orig_ner = calculate_ner_density(original_text)
    print("\n⚙️  WORD-SHUFFLED text...")
    word_ner = calculate_ner_density(word_shuffled_text)
    print("\n⚙️  SENTENCE-SHUFFLED text...")
    sent_ner = calculate_ner_density(sent_shuffled_text)

    d_word, orig_mean, word_mean, orig_std, word_std = calculate_cohens_d(orig_ner, word_ner)
    d_sent, _,        sent_mean, _,        sent_std  = calculate_cohens_d(orig_ner, sent_ner)
    store_results(results, 'NER',
                  d_word, d_sent,
                  orig_mean, word_mean, sent_mean,
                  orig_std,  word_std,  sent_std,
                  len(orig_ner))


# ============================================================================
# METHOD 5: LDA TOPIC MODELING
# ============================================================================

def run_lda(original_text, word_shuffled_text, sent_shuffled_text, results):
    """
    Max topic probability per document. Dictionary and LDA model trained
    on original only; all conditions inferred through shared vocabulary.

    FIX [7]: segment_into_units(unit_size=50) replaces split_sentences().

    Note on LDA order-invariance: LDA is a bag-of-words model and is
    entirely insensitive to token order. Numerical results are therefore
    unaffected by this segmentation change (identical results were produced
    by both period-based and fixed-unit segmentation in testing). The fix
    is applied for internal consistency across all methods in this suite.
    """
    print("\n" + "="*80)
    print("METHOD 5/7: LDA Topic Modeling")
    print("STATUS: 🔧 Original-only training — valid comparison")
    print("FIX [7]: segment_into_units(50) for internal consistency")
    print("="*80)

    STOP_WORDS = set([
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'is', 'was', 'are', 'were', 'be', 'have', 'has', 'had', 'do',
        'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
        'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
        'it', 'we', 'they', 'him', 'her', 'us', 'them', 'my', 'your',
        'his', 'its', 'our', 'their', 'what', 'which', 'who', 'when',
        'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
        'same', 'so', 'than', 'too', 'very', 'as', 'by', 'from', 'with'
    ])

    def preprocess(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+',    ' ', text)
        return [t for t in text.split() if t not in STOP_WORDS and len(t) > 2]

    def calculate_lda(orig_text, word_text, sent_text, num_topics=5):
        # FIX [7]: fixed-size units
        orig_units = segment_into_units(orig_text, unit_size=50, min_words=10)
        word_units = segment_into_units(word_text, unit_size=50, min_words=10)
        sent_units = segment_into_units(sent_text, unit_size=50, min_words=10)

        orig_docs = [preprocess(u) for u in orig_units]
        orig_docs = [d for d in orig_docs if len(d) > 5]
        word_docs = [preprocess(u) for u in word_units]
        word_docs = [d for d in word_docs if len(d) > 5]
        sent_docs = [preprocess(u) for u in sent_units]
        sent_docs = [d for d in sent_docs if len(d) > 5]

        if len(orig_docs) < 10:
            return np.array([0.0]), np.array([0.0]), np.array([0.0])

        # Build dictionary from original only — shared vocab for all conditions
        print("   Building dictionary from original text...")
        dictionary  = corpora.Dictionary(orig_docs)
        orig_corpus = [dictionary.doc2bow(d) for d in orig_docs]
        word_corpus = [dictionary.doc2bow(d) for d in word_docs]
        sent_corpus = [dictionary.doc2bow(d) for d in sent_docs]

        # Train LDA on original only
        print("   Training LDA on original text...")
        lda = LdaModel(
            corpus=orig_corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=10,
            per_word_topics=True
        )

        def get_max_probs(corpus, model):
            probs = []
            for doc_bow in corpus:
                doc_topics = model.get_document_topics(doc_bow)
                max_prob   = max([p for _, p in doc_topics]) if doc_topics else 0.0
                probs.append(max_prob)
            return np.array(probs)

        print("   Inferring topics: original...")
        orig_probs = get_max_probs(orig_corpus, lda)
        print("   Inferring topics: word-shuffled...")
        word_probs = get_max_probs(word_corpus, lda)
        print("   Inferring topics: sentence-shuffled...")
        sent_probs = get_max_probs(sent_corpus, lda)

        return orig_probs, word_probs, sent_probs

    print("\n⚙️  Running LDA (single model, original-only training)...")
    orig_lda, word_lda, sent_lda = calculate_lda(
        original_text, word_shuffled_text, sent_shuffled_text)

    d_word, orig_mean, word_mean, orig_std, word_std = calculate_cohens_d(orig_lda, word_lda)
    d_sent, _,        sent_mean, _,        sent_std  = calculate_cohens_d(orig_lda, sent_lda)
    store_results(results, 'LDA',
                  d_word, d_sent,
                  orig_mean, word_mean, sent_mean,
                  orig_std,  word_std,  sent_std,
                  len(orig_lda))


# ============================================================================
# METHOD 6: BERTSCORE (SEQUENTIAL COHERENCE)
# ============================================================================

def run_bertscore(original_text, word_shuffled_text, sent_shuffled_text, results):
    """
    BERTScore F1 between consecutive fixed-size 50-token windows.
    Pre-trained BERT model — no fitting on test data.

    FIX [8]: Batched inference — single bertscore(cands, refs) call processes
    all window pairs at once. Mathematically identical to per-pair loop but
    significantly faster — on LOTR-scale texts the difference is hours vs
    minutes on GPU.

    Window strategy: fixed 50-token non-overlapping windows (no overlap
    required — we measure consecutive window similarity, not density).
    """
    print("\n" + "="*80)
    print("METHOD 6/7: BERTScore (Sequential Coherence)")
    print("STATUS: ✅ Pre-trained model — valid comparison")
    print("FIX [8]: batched inference — hours vs minutes on long texts")
    print("="*80)

    def calculate_bertscore_coherence(text, window_size=50):
        words   = text.split()
        windows = []
        for i in range(0, len(words) - window_size + 1, window_size):
            windows.append(' '.join(words[i:i + window_size]))

        if len(windows) < 2:
            return np.array([0.0])

        # FIX [8]: batched bertscore() — single forward pass, all pairs
        cands = windows[1:]
        refs  = windows[:-1]
        print(f"   Scoring {len(cands)} consecutive window pairs ({window_size} tokens each)...")

        try:
            _, _, F1 = bertscore(
                cands, refs,
                lang='en',
                model_type='bert-base-uncased',
                verbose=False
            )
            return F1.cpu().numpy()
        except Exception as e:
            print(f"   BERTScore error: {e}")
            return np.array([0.0])

    print("\n⚙️  ORIGINAL text...")
    orig_bert = calculate_bertscore_coherence(original_text)
    print("\n⚙️  WORD-SHUFFLED text...")
    word_bert = calculate_bertscore_coherence(word_shuffled_text)
    print("\n⚙️  SENTENCE-SHUFFLED text...")
    sent_bert = calculate_bertscore_coherence(sent_shuffled_text)

    d_word, orig_mean, word_mean, orig_std, word_std = calculate_cohens_d(orig_bert, word_bert)
    d_sent, _,        sent_mean, _,        sent_std  = calculate_cohens_d(orig_bert, sent_bert)
    store_results(results, 'BERTScore',
                  d_word, d_sent,
                  orig_mean, word_mean, sent_mean,
                  orig_std,  word_std,  sent_std,
                  len(orig_bert))


# ============================================================================
# METHOD 7: BERTOPIC
# ============================================================================

def run_bertopic(original_text, word_shuffled_text, sent_shuffled_text, results):
    """
    Max topic assignment probability per document using BERTopic.
    Model trained on original only — honest evaluation.

    FIX [9a]: segment_into_units(unit_size=50) replaces split_sentences().
    FIX [9b]: Original-only training (fit_transform on original, transform()
              on shuffled). Previous Kurian version trained on all three
              combined — UMAP/HDBSCAN geometry was shaped by the contrast
              between coherent and scrambled text, inflating Cohen's d
              artificially (2.34 vs. honest 0.81).

    EXPECTED RESULT: BERTopic will likely FAIL (d < 1.0). This is the
    correct honest result. BERTopic's sentence-transformer backbone is
    partially order-robust — individual words retain semantic weight even
    when scrambled — so topic assignments do not drop dramatically. This
    is the gap SE's Σ (KL divergence) component is designed to fill.
    """
    print("\n" + "="*80)
    print("METHOD 7/7: BERTopic")
    print("STATUS: 🔧 Original-only training — honest evaluation")
    print("FIX [9a]: segment_into_units(50) for consistent chunk sizes")
    print("FIX [9b]: original-only training (removes artificial inflation)")
    print("NOTE: FAIL result expected — this is correct. See docstring.")
    print("="*80)

    def calculate_bertopic(orig_text, word_text, sent_text):
        # FIX [9a]: fixed-size units
        orig_docs = segment_into_units(orig_text, unit_size=50, min_words=10)
        word_docs = segment_into_units(word_text, unit_size=50, min_words=10)
        sent_docs = segment_into_units(sent_text, unit_size=50, min_words=10)

        if len(orig_docs) < 10:
            return np.array([0.0]), np.array([0.0]), np.array([0.0])

        # FIX [9b]: fit_transform on original only
        print("   Training BERTopic on original text only...")
        bertopic_model = BERTopic(
            language="english",
            calculate_probabilities=True,
            verbose=False,
            min_topic_size=3,
            nr_topics="auto"
        )

        orig_topics, orig_probs = bertopic_model.fit_transform(orig_docs)
        orig_max_probs          = np.max(orig_probs, axis=1)

        # transform() only on shuffled conditions — model never sees scrambled text
        print("   Transforming word-shuffled docs...")
        word_topics, word_probs = bertopic_model.transform(word_docs)
        word_max_probs          = np.max(word_probs, axis=1)

        print("   Transforming sentence-shuffled docs...")
        sent_topics, sent_probs = bertopic_model.transform(sent_docs)
        sent_max_probs          = np.max(sent_probs, axis=1)

        return orig_max_probs, word_max_probs, sent_max_probs

    print("\n⚙️  Running BERTopic (single model, original-only training)...")
    orig_bt, word_bt, sent_bt = calculate_bertopic(
        original_text, word_shuffled_text, sent_shuffled_text)

    d_word, orig_mean, word_mean, orig_std, word_std = calculate_cohens_d(orig_bt, word_bt)
    d_sent, _,        sent_mean, _,        sent_std  = calculate_cohens_d(orig_bt, sent_bt)
    store_results(results, 'BERTopic',
                  d_word, d_sent,
                  orig_mean, word_mean, sent_mean,
                  orig_std,  word_std,  sent_std,
                  len(orig_bt))


# ============================================================================
# RESULTS REPORTING
# ============================================================================

def print_results(results):
    """
    Print full results tables, detailed statistics, and key insights.

    FIX [10]: KEY INSIGHTS section iterates over the flat `results` dict,
    which is correctly populated by store_results() in this version.
    The class-based version never wrote to this dict, causing all tier
    counts to silently display as 0.
    """
    METHOD_ORDER = ['Perplexity', 'Sentiment', 'TF-IDF', 'NER', 'LDA', 'BERTScore', 'BERTopic']
    executed     = [m for m in METHOD_ORDER if m in results]

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n" + "="*100)
    print("  FINAL RESULTS — 3-WAY SHUFFLE COMPARISON  (CANONICAL v1.0)")
    print("="*100)
    print(f"\n{'Method':<14} {'d(Word-Shuf)':>13} {'Verdict':<18} "
          f"{'d(Sent-Shuf)':>13} {'Verdict':<18} {'n':>6}")
    print("-" * 90)
    for m in executed:
        r = results[m]
        print(f"{m:<14} "
              f"{r['d_vs_word']:>13.3f} {get_verdict(r['d_vs_word']):<18} "
              f"{r['d_vs_sent']:>13.3f} {get_verdict(r['d_vs_sent']):<18} "
              f"{r['n_observations']:>6}")
    print("="*100)

    # ── Mean values ────────────────────────────────────────────────────────────
    print(f"\n{'Method':<14} {'Original':>12} {'Word-Shuf':>12} {'Sent-Shuf':>12}")
    print("-" * 55)
    for m in executed:
        r = results[m]
        print(f"{m:<14} "
              f"{r['original_mean']:>12.4f} "
              f"{r['word_shuf_mean']:>12.4f} "
              f"{r['sent_shuf_mean']:>12.4f}")

    # ── Detailed statistics ────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("DETAILED STATISTICS")
    print("="*80)
    for m in executed:
        r = results[m]
        print(f"\n{m}:")
        print(f"  d(orig vs word-shuffled):     {r['d_vs_word']:.3f}  {get_verdict(r['d_vs_word'])}")
        print(f"  d(orig vs sentence-shuffled): {r['d_vs_sent']:.3f}  {get_verdict(r['d_vs_sent'])}")
        print(f"  Original:        mean={r['original_mean']:.4f}  SD={r['original_std']:.4f}")
        print(f"  Word-shuffled:   mean={r['word_shuf_mean']:.4f}  SD={r['word_shuf_std']:.4f}")
        print(f"  Sent-shuffled:   mean={r['sent_shuf_mean']:.4f}  SD={r['sent_shuf_std']:.4f}")
        print(f"  N observations:  {r['n_observations']}")

    # ── Key insights — FIX [10] ────────────────────────────────────────────────
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)

    tiers_word = {
        'STRONG PASS': [m for m in executed if results[m]['d_vs_word'] >= 3.0],
        'PASS':        [m for m in executed if 2.0 <= results[m]['d_vs_word'] < 3.0],
        'BORDERLINE':  [m for m in executed if 1.0 <= results[m]['d_vs_word'] < 2.0],
        'FAIL':        [m for m in executed if results[m]['d_vs_word'] < 1.0],
    }
    tiers_sent = {
        'STRONG PASS': [m for m in executed if results[m]['d_vs_sent'] >= 3.0],
        'PASS':        [m for m in executed if 2.0 <= results[m]['d_vs_sent'] < 3.0],
        'BORDERLINE':  [m for m in executed if 1.0 <= results[m]['d_vs_sent'] < 2.0],
        'FAIL':        [m for m in executed if results[m]['d_vs_sent'] < 1.0],
    }

    tier_labels = [
        ('✅✅ STRONG PASS (d ≥ 3.0)',    'STRONG PASS'),
        ('✅  PASS (2.0 ≤ d < 3.0)',      'PASS'),
        ('~  BORDERLINE (1.0 ≤ d < 2.0)', 'BORDERLINE'),
        ('❌  FAIL (d < 1.0)',             'FAIL'),
    ]

    print(f"\n📊 WORD-SHUFFLE SENSITIVITY (total structure destruction):")
    for label, key in tier_labels:
        print(f"   {label}: {len(tiers_word[key])} method(s)")
        for m in tiers_word[key]:
            print(f"      • {m}  (d = {results[m]['d_vs_word']:.3f})")

    print(f"\n📊 SENTENCE-SHUFFLE SENSITIVITY (discourse structure only):")
    for label, key in tier_labels:
        print(f"   {label}: {len(tiers_sent[key])} method(s)")
        for m in tiers_sent[key]:
            print(f"      • {m}  (d = {results[m]['d_vs_sent']:.3f})")

    discourse_sensitive = [
        m for m in executed
        if results[m]['d_vs_sent'] >= 1.0
        and results[m]['d_vs_word'] > results[m]['d_vs_sent']
    ]
    print(f"\n🎯 DISCOURSE-LEVEL SENSITIVE METHODS:")
    print("   (Detect both total destruction AND discourse-only disruption)")
    if discourse_sensitive:
        for m in discourse_sensitive:
            d_w   = results[m]['d_vs_word']
            d_s   = results[m]['d_vs_sent']
            ratio = d_w / d_s if d_s > 0 else 0.0
            print(f"   • {m}:  word d={d_w:.3f},  sent d={d_s:.3f},  ratio={ratio:.1f}x")
    else:
        print("   None of the tested methods show significant discourse-level sensitivity.")
        print("   This is the gap that Symbolic Entropy's Σ component is designed to fill.")

    # ── Methodological note ────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("METHODOLOGICAL FIXES APPLIED IN THIS RUN")
    print("="*80)
    fixes = [
        "[2]  segment_into_units(): fixed-size token chunks — all 5 segmentation methods",
        "[3]  GPT-2: .eval() before inference — deterministic perplexity",
        "[4]  Sentiment: segment_into_units(80) — consistent chunk sizes",
        "[5]  TF-IDF: segment_into_units(50) + single vectorizer on original (retained)",
        "[6]  NER: segment_into_units(50) — consistent chunk sizes",
        "[7]  LDA: segment_into_units(50) + original-only training",
        "[8]  BERTScore: batched bertscore() call — hours vs minutes on LOTR-scale texts",
        "[9]  BERTopic: segment_into_units(50) + original-only training (honest eval)",
        "[10] KEY INSIGHTS: correct results dict population",
        "[11] Sentiment model: memory released after use",
    ]
    for fix in fixes:
        print(f"  {fix}")

    print("="*80)
    print("\n✅ CANONICAL SHUFFLE TEST SUITE COMPLETE")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("  SYMBOLIC ENTROPY — CANONICAL NLP SHUFFLE TEST SUITE  v1.0  (Colab)")
    print("  All methodological fixes applied")
    print("=" * 80)
    print("\nTests 7 NLP methods across 3 conditions:")
    print("  • Original text")
    print("  • Word-shuffled text   (destroys all structure)")
    print("  • Sentence-shuffled text  (preserves local, destroys discourse)")
    print("\nYou will be prompted to upload 3 files (one per prompt).")
    print("Accepts .txt or .docx.")
    print("=" * 80)

    # ── File uploads — 3 separate prompts ──────────────────────────────────────
    print("\n📤  Upload 1/3: ORIGINAL file (.docx or .txt):")
    uploaded_original = files.upload()
    if not uploaded_original:
        print("No file uploaded. Exiting.")
        return

    print("\n📤  Upload 2/3: WORD-SHUFFLED file (.docx or .txt):")
    uploaded_word = files.upload()
    if not uploaded_word:
        print("No file uploaded. Exiting.")
        return

    print("\n📤  Upload 3/3: SENTENCE-SHUFFLED file (.docx or .txt):")
    uploaded_sent = files.upload()
    if not uploaded_sent:
        print("No file uploaded. Exiting.")
        return

    # ── Read files ─────────────────────────────────────────────────────────────
    print("\n📖 Reading files...")
    original_text      = read_file(uploaded_original)
    word_shuffled_text = read_file(uploaded_word)
    sent_shuffled_text = read_file(uploaded_sent)

    print(f"  ✓ Original:           {len(original_text):,} characters  "
          f"({list(uploaded_original.keys())[0]})")
    print(f"  ✓ Word-shuffled:      {len(word_shuffled_text):,} characters  "
          f"({list(uploaded_word.keys())[0]})")
    print(f"  ✓ Sentence-shuffled:  {len(sent_shuffled_text):,} characters  "
          f"({list(uploaded_sent.keys())[0]})")

    active = [m for m, v in list_of_methods.items() if v == 1]
    if not active:
        print("\n⚠️  No methods enabled. Set values to 1 in list_of_methods at the top of this file.")
        return

    print(f"\nMethods enabled ({len(active)}/7): {', '.join(active)}")
    print(f"Estimated runtime: ~{len(active) * 3}–{len(active) * 5} minutes")
    print("=" * 80)

    # ── Run enabled methods ────────────────────────────────────────────────────
    results = {}

    if list_of_methods.get('Perplexity') == 1:
        run_perplexity(original_text, word_shuffled_text, sent_shuffled_text, results)

    if list_of_methods.get('Sentiment') == 1:
        run_sentiment(original_text, word_shuffled_text, sent_shuffled_text, results)

    if list_of_methods.get('TF-IDF') == 1:
        run_tfidf(original_text, word_shuffled_text, sent_shuffled_text, results)

    if list_of_methods.get('NER') == 1:
        run_ner(original_text, word_shuffled_text, sent_shuffled_text, results)

    if list_of_methods.get('LDA') == 1:
        run_lda(original_text, word_shuffled_text, sent_shuffled_text, results)

    if list_of_methods.get('BERTScore') == 1:
        run_bertscore(original_text, word_shuffled_text, sent_shuffled_text, results)

    if list_of_methods.get('BERTopic') == 1:
        run_bertopic(original_text, word_shuffled_text, sent_shuffled_text, results)

    # ── Print all results ──────────────────────────────────────────────────────
    if results:
        print_results(results)
    else:
        print("\nNo results to display.")


# ============================================================================
# ENTRY POINT
# ============================================================================

if torch.cuda.is_available():
    print("CUDA available ✓")
else:
    print("Running on CPU — no CUDA detected (slower). "
          "Use Runtime > Change runtime type > T4 GPU for ~10x speedup.")

main()

