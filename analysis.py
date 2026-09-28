import argparse
import ast
import csv
import hashlib
import io
import itertools
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import tokenize
import wave
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
sys.dont_write_bytecode = True

import numpy as np
import scipy
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parent


def require(condition, code):
    if not condition:
        raise ValueError(code)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8388608), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_features(path, features):
    buffer = io.BytesIO()
    np.save(buffer, features, allow_pickle=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=6) as zipped:
        member = ZipInfo("features.npy", (1980, 1, 1, 0, 0, 0))
        member.compress_type = ZIP_DEFLATED
        member.external_attr = 0o100644 << 16
        zipped.writestr(member, buffer.getvalue())
    temporary.replace(path)


def parse_textgrid(path):
    text = path.read_text(encoding="utf-8-sig")
    require('Object class = "TextGrid"' in text, "E_TEXTGRID_FORMAT")
    tiers = {}
    for part in re.split(r"(?m)^\s*item \[\d+\]:\s*$", text)[1:]:
        name = re.search(r'name = "((?:[^"\n]|"")*)"', part)
        size = re.search(r"intervals: size = (\d+)", part)
        require(name is not None and size is not None, "E_INTERVAL_TIER")
        matches = re.findall(r'intervals \[(\d+)\]:\s*xmin = ([\d.Ee+-]+)\s*xmax = ([\d.Ee+-]+)\s*text = "((?:[^"\n]|"")*)"', part)
        require(len(matches) == int(size.group(1)), "E_INTERVAL_COUNT")
        key = name.group(1).replace('""', '"')
        require(key not in tiers, "E_DUPLICATE_TIER")
        intervals = []
        for expected, (index, start, end, label) in enumerate(matches, 1):
            require(int(index) == expected, "E_INTERVAL_ORDER")
            a, b = round(float(start), 5), round(float(end), 5)
            require(math.isfinite(a) and math.isfinite(b) and 0 <= a <= b, "E_INTERVAL_BOUNDS")
            intervals.append((a, b, label.replace('""', '"').strip()))
        tiers[key] = intervals
    require({"ORT-MAU", "KAN-MAU", "MAU"}.issubset(tiers), "E_REQUIRED_TIERS")
    require(len(tiers["ORT-MAU"]) == len(tiers["KAN-MAU"]), "E_WORD_TIERS")
    return tiers


@dataclass
class Corpus:
    root: Path
    metadata: list
    rows: list
    valid: np.ndarray
    classes: list
    speakers: np.ndarray
    words: np.ndarray
    labels: np.ndarray
    contexts: np.ndarray
    y: np.ndarray
    records: list
    weights: np.ndarray
    population: dict
    excluded_without_archived_frames: int


def load_corpus(root=ROOT):
    root = Path(root).resolve()
    with (root / "data/metadata.csv").open(encoding="utf-8", newline="") as stream:
        metadata = list(csv.DictReader(stream))
    require(len(metadata) == 37, "E_COHORT_SIZE")
    names = [row["filename"] for row in metadata]
    require(len(set(names)) == len(names), "E_COHORT_DUPLICATES")
    require(names == sorted(names, key=lambda name: name + "_brazil.csv"), "E_FOLD_ORDER")
    require(all(re.fullmatch(r"portuguese[0-9]+", name) for name in names), "E_COHORT_NAMES")
    require({p.stem for p in (root / "data/textgrids").glob("*.TextGrid")} == set(names), "E_TEXTGRID_COHORT")
    rows = []
    skipped = 0
    for speaker, metadata_row in enumerate(metadata):
        require(metadata_row["country"] == "brazil" and metadata_row["native_language"] == "portuguese", "E_COHORT_ORIGIN")
        for field in ("sample_rate", "channels", "samples", "sample_width"):
            metadata_row[field] = int(metadata_row[field])
        require(metadata_row["sample_rate"] == 44100 and metadata_row["channels"] == 1, "E_AUDIO_GRID")
        require(metadata_row["samples"] > 0, "E_AUDIO_LENGTH")
        tiers = parse_textgrid(root / "data/textgrids" / (metadata_row["filename"] + ".TextGrid"))
        phones = tiers["MAU"]
        words = tiers["ORT-MAU"]
        times = (np.arange(1 + metadata_row["samples"] // 441) * 441 + 1024) / 44100
        for index, (start, end, label) in enumerate(phones):
            if label == "<p:>" or end <= start:
                continue
            selected = np.flatnonzero((times >= start) & (times < end))
            if not len(selected):
                skipped += 1
                continue
            midpoint = .5 * (start + end)
            word = next((text for a, b, text in words if a <= midpoint < b), "").strip().casefold()
            require(bool(word) and bool(label), "E_EMPTY_LABEL")
            rows.append({"speaker": speaker, "word": word, "label": label,
                         "left": phones[index - 1][2] if index else "",
                         "right": phones[index + 1][2] if index + 1 < len(phones) else "",
                         "start": start, "end": end, "segment_index": index,
                         "archived_first": int(selected[0]), "archived_last": int(selected[-1])})
    class_words = defaultdict(set)
    for row in rows:
        class_words[row["label"]].add(row["word"])
    classes = sorted(label for label, words in class_words.items() if len(words) >= 2)
    valid = np.array([row["label"] in classes for row in rows])
    eligible = [row for row, keep in zip(rows, valid) if keep]
    speakers = np.array([row["speaker"] for row in eligible], dtype=int)
    words = np.array([row["word"] for row in eligible])
    labels = np.array([row["label"] for row in eligible])
    contexts = np.array([json.dumps((row["left"], row["right"]), ensure_ascii=True) for row in eligible])
    y = np.searchsorted(classes, labels)
    records = [(row["speaker"], row["word"], row["label"], row["left"], row["right"]) for row in eligible]
    weights = np.zeros(len(y))
    for speaker in range(len(metadata)):
        for label in range(len(classes)):
            mask = (speakers == speaker) & (y == label)
            require(mask.any(), "E_MACRO_CELL")
            weights[mask] = 1 / (len(metadata) * len(classes) * mask.sum())
    require(abs(weights.sum() - 1) < 1e-12, "E_MACRO_WEIGHT")
    population = {"pre_eligibility_tokens": len(rows), "all_labels": len(class_words),
                  "excluded_tokens": int((~valid).sum()),
                  "one_word_labels": sorted(set(class_words) - set(classes)),
                  "eligible_tokens": len(eligible), "classes": len(classes),
                  "speakers": len(metadata), "words": len(set(words))}
    return Corpus(root, metadata, rows, valid, classes, speakers, words, labels,
                  contexts, y, records, weights, population, skipped)


def load_arrays(corpus, directory=None):
    directory = Path(directory) if directory else corpus.root / "data/features"
    require({p.stem for p in directory.glob("*.npz")} == {row["filename"] for row in corpus.metadata}, "E_FEATURE_COHORT")
    arrays = []
    for row in corpus.metadata:
        with np.load(directory / (row["filename"] + ".npz"), allow_pickle=False) as zipped:
            require(zipped.files == ["features"], "E_FEATURE_KEYS")
            array = zipped["features"]
        require(array.dtype == np.float32 and array.shape == (1 + row["samples"] // 441, 39, 2), "E_FEATURE_SHAPE")
        require(np.isfinite(array).all(), "E_NONFINITE_FEATURES")
        arrays.append(array)
    return arrays


def central_indices(indices, fraction=.6):
    n = len(indices)
    require(n > 0 and 0 < fraction <= 1, "E_CENTRAL_WINDOW")
    trim = (1 - fraction) / 2
    lo = min(n - 1, max(0, round(n * trim)))
    hi = max(lo + 1, min(n, round(n * (1 - trim))))
    return indices[lo:hi]


def represent(corpus, arrays, fraction=.6, block="all", mapping="centered"):
    require(block in ("all", "static") and mapping in ("centered", "archived"), "E_REPRESENTATION")
    centers = [np.arange(len(array)) * 441 / 44100 for array in arrays]
    vectors = []
    for row in corpus.rows:
        speaker = row["speaker"]
        if mapping == "archived":
            indices = np.arange(row["archived_first"], row["archived_last"] + 1)
        else:
            indices = np.flatnonzero((centers[speaker] >= row["start"]) & (centers[speaker] < row["end"]))
        indices = central_indices(indices, fraction)
        vector = np.asarray(arrays[speaker][indices[0]:indices[-1] + 1, :, 1], dtype=float).mean(axis=0)
        if block == "static":
            vector = vector[:13]
        vectors.append(vector)
    result = np.stack(vectors)[corpus.valid]
    require(np.isfinite(result).all(), "E_VECTOR_FINITE")
    return result


def audit_timing(corpus, arrays):
    counts = Counter()
    residuals = [0., 0.]
    for array in arrays:
        for order in (1, 2):
            derivative = savgol_filter(array[:, :13, 0].astype(float), 9, order, deriv=order, axis=0, mode="interp")
            residuals[order - 1] = max(residuals[order - 1], float(np.abs(derivative - array[:, 13 * order:13 * (order + 1), 0]).max()))
    centers = [np.arange(len(array)) * 441 / 44100 for array in arrays]
    for row in corpus.rows:
        times = centers[row["speaker"]]
        inherited = times + 1024 / 44100
        old = np.flatnonzero((inherited >= row["start"]) & (inherited < row["end"]))
        corrected = np.flatnonzero((times >= row["start"]) & (times < row["end"]))
        require(len(old) and len(corrected), "E_AUDIT_EMPTY_INTERVAL")
        counts["phone_rows"] += 1
        counts["archived_intervals_match_offset_formula"] += int(old[0] == row["archived_first"] and old[-1] == row["archived_last"])
        selected = central_indices(np.arange(row["archived_first"], row["archived_last"] + 1))
        selected_corrected = central_indices(corrected)
        counts["different_central_frame_sets"] += int(not np.array_equal(selected, selected_corrected))
        counts["central_selected_frames"] += len(selected)
        outside = (times[selected] < row["start"]) | (times[selected] >= row["end"])
        counts["central_frames_centers_outside_phone"] += int(outside.sum())
        counts["tokens_any_selected_frame_center_outside_phone"] += int(outside.any())
        crosses = (times[selected] - .04 < row["start"]) | (times[selected] + .04 >= row["end"])
        counts["tokens_any_dynamic_context_outside_phone"] += int(crosses.any())
        counts["tokens_all_dynamic_contexts_reach_outside_phone"] += int(crosses.all())
        radius = .04 + 1024 / 44100
        counts["tokens_all_dynamic_and_window_supports_reach_outside_phone"] += int(((times[selected] - radius < row["start"]) | (times[selected] + radius >= row["end"])).all())
    return {"counts_before_eligibility": dict(counts), "recorded_sample_rates": sorted({row["sample_rate"] for row in corpus.metadata}),
            "feature_arrays_checked": len(arrays), "audio_headers_checked_in_this_run": 0,
            "delta_formula_max_absolute_residual": residuals, "n_fft": 2048, "hop": 441,
            "delta_width": 9, "textgrid_decimal_places": 5,
            "excluded_without_archived_frames": corpus.excluded_without_archived_frames}


PARTITIONS = {'class': (2,), 'class_speaker': (2, 0), 'class_left': (2, 3), 'class_right': (2, 4), 'class_both': (2, 3, 4), 'class_left_speaker': (2, 3, 0), 'class_right_speaker': (2, 4, 0), 'class_both_speaker': (2, 3, 4, 0)}

REFINEMENTS = (('class', 'class_speaker'), ('class', 'class_left'), ('class', 'class_right'), ('class_left', 'class_both'), ('class_right', 'class_both'), ('class_left', 'class_left_speaker'), ('class_right', 'class_right_speaker'), ('class_both', 'class_both_speaker'), ('class_speaker', 'class_both_speaker'))

def synthetic_checks():
    cases = 0
    for n in range(1, 11):
        for r in range(n + 1):
            k = n - r
            occupancies = [sum((i < r for i in chosen)) for chosen in itertools.combinations(range(n), k)]
            e = Fraction(sum(occupancies), len(occupancies))
            u = max(occupancies)
            require(e == Fraction(r * k, n), 'E_UNIFORM_EXPECTATION_FAILED')
            require(set(occupancies) == set(range(min(r, k) + 1)), 'E_FEASIBLE_RANGE_FAILED')
            require(occupancies.count(0) == 1, 'E_ZERO_SUBSET_UNIQUENESS_FAILED')
            require(Fraction(occupancies.count(0), len(occupancies)) == Fraction(1, math.comb(n, r)), 'E_ZERO_EVENT_PROBABILITY_FAILED')
            require(Fraction(u, 2) <= e <= u, 'E_FACTOR_TWO_BOUND_FAILED')
            cases += 1
    refinement_cases = 0
    for n1, n2 in itertools.product(range(1, 9), repeat=2):
        for r1, r2 in itertools.product(range(n1 + 1), range(n2 + 1)):
            n, r = (n1 + n2, r1 + r2)
            coarse = Fraction(r * (n - r), n)
            fine = Fraction(r1 * (n1 - r1), n1) + Fraction(r2 * (n2 - r2), n2)
            between = n1 * (Fraction(r1, n1) - Fraction(r, n)) ** 2
            between += n2 * (Fraction(r2, n2) - Fraction(r, n)) ** 2
            require(coarse - fine == between, 'E_REFINEMENT_IDENTITY_FAILED')
            require(min(r1, n1 - r1) + min(r2, n2 - r2) <= min(r, n - r), 'E_CAPACITY_REFINEMENT_FAILED')
            refinement_cases += 1
    return {'status': 'PASS_FINITE_EXACT_FIXTURES', 'single_block_cases': cases, 'two_block_refinements': refinement_cases, 'low_density_high_presence_example': {'associated_records': 1000, 'other_records': 1, 'retained_slots': 1, 'maximum_retained_fraction': 0.001, 'expected_retained_fraction': float(Fraction(1, 1001)), 'probability_any_associated_record': float(Fraction(1000, 1001))}}

def distribution(values):
    values = sorted(values)
    n = len(values)
    require(n > 0, 'E_EMPTY_DESCRIPTIVE_DISTRIBUTION')
    return {'min': float(values[0]), 'median': float((values[(n - 1) // 2] + values[n // 2]) / 2), 'max': float(values[-1]), 'mean': float(sum(values, Fraction()) / n)}

def analyze(records):
    people = {r[0] for r in records}
    classes = {r[2] for r in records}
    class_counts = Counter(((r[0], r[2]) for r in records))
    require(len(class_counts) == len(people) * len(classes), 'E_INCOMPLETE_MACRO_TEST_GRID')
    test_cells = defaultdict(list)
    for row in records:
        test_cells[row[:2]].append(row)
    weights = {key: sum((Fraction(1, len(people) * len(classes) * class_counts[r[0], r[2]]) for r in test), Fraction()) for key, test in test_cells.items()}
    require(sum(weights.values()) == 1, 'E_TEST_MACRO_WEIGHTS_DO_NOT_SUM_TO_ONE')
    summaries, private_cells = ({}, {})
    for name, fields in PARTITIONS.items():
        all_counts = Counter()
        speaker_counts, word_counts, joint_counts = (defaultdict(Counter) for _ in range(3))
        for row in records:
            speaker, word = row[:2]
            block = tuple((row[i] for i in fields))
            all_counts[block] += 1
            speaker_counts[speaker][block] += 1
            word_counts[word][block] += 1
            joint_counts[speaker, word][block] += 1
        cells = {}
        for key, test in test_cells.items():
            speaker, word = key
            R, U, E, zero_denominator = (0, 0, Fraction(), 1)
            for block, total_r in word_counts[word].items():
                r = total_r - joint_counts[key][block]
                n = all_counts[block] - speaker_counts[speaker][block]
                require(0 <= r <= n, 'E_INVALID_HELD_SPEAKER_SUBTRACTION')
                if not r:
                    continue
                k = n - r
                R += r
                U += min(r, k)
                E += Fraction(r * k, n)
                zero_denominator *= math.comb(n, r)
            require(R > 0 and E <= U <= 2 * E, 'E_INVALID_SUPPORT_BOUNDS')
            P = 1 - Fraction(1, zero_denominator)
            require((P == 0) == (U == 0), 'E_AVAILABILITY_DISAGREES_WITH_FEASIBILITY')
            cells[key] = {'R': R, 'U': U, 'E': E, 'P': P, 'weight': weights[key], 'test_tokens': len(test)}
        private_cells[name] = cells
        all_cells = list(cells.values())
        variable = [c for c in all_cells if c['U'] > 0]
        forced = [c for c in all_cells if c['U'] == 0]
        R = sum((c['R'] for c in all_cells))
        E = sum((c['E'] for c in all_cells), Fraction())
        U = sum((c['U'] for c in all_cells))
        variable_R = sum((c['R'] for c in variable))
        possible_weight = sum((c['weight'] for c in variable), Fraction())
        probability_weight = sum((c['weight'] * c['P'] for c in all_cells), Fraction())
        summaries[name] = {'cells': len(cells), 'forced_cells': len(forced), 'forced_test_tokens': sum((c['test_tokens'] for c in forced)), 'forced_macro_weight': float(1 - possible_weight), 'possible_macro_weight': float(possible_weight), 'uniform_macro_weighted_any_associated_segment_probability': float(probability_weight), 'uniform_macro_weighted_variable_zero_probability': float(possible_weight - probability_weight), 'nonforced_any_probability_distribution': distribution([c['P'] for c in variable]), 'nonforced_macro_weighted_any_probability': float(probability_weight / possible_weight), 'training_incidence_totals': {'held': R, 'maximum': U, 'expected': float(E), 'nonforced_held': variable_R}, 'expected_held_retention_fraction': float(E / R), 'sharp_maximum_held_retention_fraction': U / R, 'nonforced_expected_held_retention_fraction': float(E / variable_R), 'nonforced_maximum_held_retention_fraction': U / variable_R, 'test_macro_weighted_expected_fraction': float(sum((c['weight'] * c['E'] / c['R'] for c in all_cells), Fraction())), 'test_macro_weighted_maximum_fraction': float(sum((c['weight'] * Fraction(c['U'], c['R']) for c in all_cells), Fraction()))}
    comparisons = 0
    for coarse, fine in REFINEMENTS:
        for key in test_cells:
            a, b = (private_cells[coarse][key], private_cells[fine][key])
            require(a['U'] >= b['U'] and a['E'] >= b['E'], 'E_REFINEMENT_INCREASED_DENSITY')
            comparisons += 1
    graph = defaultdict(set)
    for _, word, label, left, right in records:
        graph[label, left, right].add(word)
    return {'population': {'tokens': len(records), 'speakers': len(people), 'classes': len(classes), 'words': len({r[1] for r in records})}, 'partitions': summaries, 'refinement_comparisons': comparisons, 'context_word_support': {'contexts': len(graph), 'contexts_in_multiple_words': sum((len(w) > 1 for w in graph.values()))}}

def audit_support(speakers, words, labels, contexts, classes):
    people = sorted(set(speakers))
    y = np.searchsorted(classes, labels)
    weights = np.zeros(len(y))
    available_before_word_removal = np.zeros(len(y), dtype=bool)
    available = np.zeros(len(y), dtype=bool)
    remaining_words = np.zeros(len(y), dtype=int)
    for speaker in people:
        for label in range(len(classes)):
            mask = (speakers == speaker) & (y == label)
            require(mask.any(), 'E_MISSING_SPEAKER_CLASS_TEST_CELL')
            weights[mask] = 1 / (len(people) * len(classes) * mask.sum())
            baseline_train = (speakers != speaker) & (y == label)
            available_before_word_removal[mask] = np.isin(contexts[mask], contexts[baseline_train])
    require(abs(weights.sum() - 1) < 1e-12, 'E_MACRO_WEIGHTS_DO_NOT_SUM_TO_ONE')
    cells = []
    totals = Counter()
    minimum_training = len(y)
    for speaker in people:
        train = np.flatnonzero(speakers != speaker)
        grouped = defaultdict(list)
        for index in train:
            grouped[int(y[index]), contexts[index]].append(index)
        blocks = [np.array(indices) for indices in grouped.values()]
        for word in sorted(set(words[speakers == speaker])):
            test = (speakers == speaker) & (words == word)
            for label in range(len(classes)):
                kept = (speakers != speaker) & (words != word) & (y == label)
                require(kept.any(), 'E_EMPTY_TRAINING_CLASS_AFTER_SPEAKER_WORD_EXCLUSION')
                minimum_training = min(minimum_training, int(kept.sum()))
                target = test & (y == label)
                available[target] = np.isin(contexts[target], contexts[kept])
                remaining_words[target] = len(set(words[kept]))
            keep_all = drop_all = partial = deleted = movable_word = movable_all = retained = 0
            expected_word = log_subsets = 0.0
            for indices in blocks:
                n = len(indices)
                r = int((words[indices] == word).sum())
                deleted += r
                retained += n - r
                if r == 0:
                    keep_all += 1
                elif r == n:
                    drop_all += 1
                else:
                    partial += 1
                    movable_word += r
                    movable_all += n
                    expected_word += r * (n - r) / n
                    log_subsets += (math.lgamma(n + 1) - math.lgamma(r + 1) - math.lgamma(n - r + 1)) / math.log(10)
            cells.append({'partial_blocks': partial, 'held_word_tokens_able_to_remain': movable_word, 'all_training_tokens_in_randomizable_blocks': movable_all, 'expected_held_word_tokens_retained': expected_word, 'expected_fraction_of_retained_training_set_shared_with_lexical': 1 - expected_word / retained, 'log10_possible_retained_subsets': log_subsets, 'test_tokens': int(test.sum()), 'test_macro_weight': float(weights[test].sum()), 'exact_lexical_deletion': partial == 0})
            for key, value in [('fully_kept_block_occurrences', keep_all), ('fully_deleted_block_occurrences', drop_all), ('randomizable_block_occurrences', partial), ('held_word_training_token_occurrences', deleted), ('held_word_training_token_occurrences_able_to_remain', movable_word), ('expected_held_word_training_token_occurrences_retained', expected_word), ('all_retained_training_token_occurrences', retained)]:
                totals[key] += value
    strata = []
    for present in (False, True):
        mask = available == present
        strata.append({'context_available': present, 'tokens': int(mask.sum()), 'classes': len(set(y[mask])), 'words': len(set(words[mask])), 'speakers': len(set(speakers[mask])), 'speaker_class_cells': len(set(zip(speakers[mask].tolist(), y[mask].tolist()))), 'speaker_word_cells': len(set(zip(speakers[mask].tolist(), words[mask].tolist())))})
    class_support = []
    for index, label in enumerate(classes):
        mask = y == index
        supported = mask & available
        class_support.append({'class': label, 'tokens': int(mask.sum()), 'word_types': len(set(words[mask])), 'neighbor_context_types': len(set(contexts[mask])), 'neighbor_contexts_spanning_multiple_words': sum((len(set(words[mask & (contexts == context)])) > 1 for context in set(contexts[mask]))), 'crossword_context_available_tokens': int(supported.sum()), 'word_types_with_any_crossword_context': len(set(words[supported])), 'remaining_words_after_test_word_removed_min_max': [int(remaining_words[mask].min()), int(remaining_words[mask].max())]})
    forced = [row for row in cells if row['exact_lexical_deletion']]
    nonforced = [row for row in cells if not row['exact_lexical_deletion']]
    summaries = ['partial_blocks', 'held_word_tokens_able_to_remain', 'all_training_tokens_in_randomizable_blocks', 'expected_held_word_tokens_retained', 'expected_fraction_of_retained_training_set_shared_with_lexical', 'log10_possible_retained_subsets']
    degeneracy = {'cells': len(cells), 'block_occurrence_totals': dict(totals), 'cell_distributions': {key: distribution([row[key] for row in cells]) for key in summaries}, 'exact_lexical_deletion_cells': len(forced), 'exact_lexical_deletion_cell_fraction': len(forced) / len(cells), 'test_tokens_in_exact_deletion_cells': sum((row['test_tokens'] for row in forced)), 'macro_weight_in_exact_deletion_cells': sum((row['test_macro_weight'] for row in forced)), 'nonexact_cell_distributions': {key: distribution([row[key] for row in nonforced]) for key in summaries}, 'held_word_training_token_occurrence_fraction_able_to_remain': totals['held_word_training_token_occurrences_able_to_remain'] / totals['held_word_training_token_occurrences'], 'expected_held_word_training_token_occurrence_fraction_retained': totals['expected_held_word_training_token_occurrences_retained'] / totals['held_word_training_token_occurrences'], 'expected_retained_training_token_overlap_with_lexical': 1 - totals['expected_held_word_training_token_occurrences_retained'] / totals['all_retained_training_token_occurrences']}
    require(not np.any(available & ~available_before_word_removal), 'E_DELETING_A_WORD_UNEXPECTEDLY_INTRODUCED_CONTEXTUAL_SUPPORT')
    support = {'minimum_training_tokens_per_class': minimum_training, 'context_available_before_word_removal': int(available_before_word_removal.sum()), 'context_absent_before_word_removal': int((~available_before_word_removal).sum()), 'context_newly_lost_after_word_removal': int((available_before_word_removal & ~available).sum()), 'context_strata': strata, 'class_support': class_support, 'label_neighbor_context_types': sum((row['neighbor_context_types'] for row in class_support)), 'label_neighbor_context_types_spanning_multiple_words': sum((row['neighbor_contexts_spanning_multiple_words'] for row in class_support)), 'classes_without_crossword_context_support': sum((row['crossword_context_available_tokens'] == 0 for row in class_support)), 'tokens_with_only_one_other_training_word': int((remaining_words == 1).sum())}
    return (support, degeneracy)

def evaluate_primary(corpus, x, controls=100, seed=20260929):
    require(controls >= 2, 'E_E_CONTROLS')
    speakers, words, labels, contexts, classes, y = (corpus.speakers, corpus.words, corpus.labels, corpus.contexts, corpus.classes, corpus.y)
    people, count = (sorted(set(speakers)), len(y))
    forced = {'cells': 0, 'tokens': 0, 'macro_weight': 0.0, 'max_centroid_difference': 0.0, 'different_predictions': 0}
    baseline = np.zeros((count, 2))
    names = ('class', 'class_speaker', 'class_context')
    draws = {name: np.zeros((count, controls)) for name in names}
    rngs = {name: np.random.default_rng(seed + i) for i, name in enumerate(names)}
    weights = corpus.weights.copy()
    overlap = np.zeros(count, dtype=bool)
    for speaker in people:
        for label in range(len(classes)):
            ix = (speakers == speaker) & (y == label)
            assert ix.any()
            weights[ix] = 1 / (len(people) * len(classes) * ix.sum())
    assert abs(weights.sum() - 1) < 1e-12
    freedom = {name: Counter() for name in names}
    for pindex, person in enumerate(people):
        train = speakers != person
        class_indices = [np.flatnonzero(train & (y == c)) for c in range(len(classes))]
        base = np.stack([x[ix].mean(axis=0) for ix in class_indices])
        blocks = {}
        for name in names:
            blocks[name] = []
            for c, ix in enumerate(class_indices):
                if name == 'class':
                    blocks[name].append((c, ix))
                else:
                    values = speakers if name == 'class_speaker' else contexts
                    for item in sorted(set(values[ix])):
                        blocks[name].append((c, ix[values[ix] == item]))
        for word in sorted(set(words[speakers == person])):
            test = np.flatnonzero((speakers == person) & (words == word))
            lexical = [ix[words[ix] != word] for ix in class_indices]
            sizes = np.array([len(ix) for ix in lexical])
            if sizes.min() < 1:
                raise ValueError('E_NONESTIMABLE_CLASS')
            held = np.stack([x[ix].mean(axis=0) for ix in lexical])
            baseline[test, 0] = cdist(x[test], base).argmin(axis=1) == y[test]
            baseline[test, 1] = cdist(x[test], held).argmin(axis=1) == y[test]
            for c in set(y[test]):
                same = test[y[test] == c]
                overlap[same] = np.isin(contexts[same], contexts[lexical[c]])
            for name in names:
                guaranteed = np.zeros_like(base)
                stochastic = []
                retained_counts = np.zeros(len(classes), dtype=int)
                for c, block in blocks[name]:
                    keep = int(np.sum(words[block] != word))
                    retained_counts[c] += keep
                    freedom[name]['blocks_evaluated'] += 1
                    if keep == 0:
                        freedom[name]['forced_empty_blocks'] += 1
                    elif keep == len(block):
                        guaranteed[c] += x[block].sum(axis=0)
                        freedom[name]['forced_full_blocks'] += 1
                    else:
                        stochastic.append((c, block, keep))
                        freedom[name]['randomizable_blocks'] += 1
                        freedom[name]['expected_test_word_tokens_retained_sum'] += float(keep * np.sum(words[block] == word) / len(block))
                assert np.array_equal(retained_counts, sizes)
                freedom[name]['speaker_word_cells'] += 1
                freedom[name]['fully_forced_cells'] += int(not stochastic)
                if name == 'class_context' and (not stochastic):
                    centers = guaranteed / sizes[:, None]
                    direct_prediction = cdist(x[test], held).argmin(axis=1)
                    grouped_prediction = cdist(x[test], centers).argmin(axis=1)
                    require(np.array_equal(direct_prediction, grouped_prediction), 'E_E_FORCED_PREDICTION')
                    selected = [np.sort(np.concatenate([block for cc, block in blocks[name] if cc == c and np.all(words[block] != word)])) for c in range(len(classes))]
                    require(all((np.array_equal(a, b) for a, b in zip(selected, lexical))), 'E_E_FORCED_SET')
                    forced['cells'] += 1
                    forced['tokens'] += len(test)
                    forced['macro_weight'] += float(weights[test].sum())
                    forced['max_centroid_difference'] = max(forced['max_centroid_difference'], float(np.abs(centers - held).max()))
                    draws[name][test] = (grouped_prediction == y[test])[:, None]
                    continue
                for replicate in range(controls):
                    sums = guaranteed.copy()
                    for c, block, keep in stochastic:
                        selected = rngs[name].choice(block, size=keep, replace=False)
                        assert len(selected) == keep and np.all(speakers[selected] != person)
                        sums[c] += x[selected].sum(axis=0)
                    centers = sums / sizes[:, None]
                    draws[name][test, replicate] = cdist(x[test], centers).argmin(axis=1) == y[test]
        if (pindex + 1) % 6 == 0:
            print(json.dumps({'stage': 'acoustic', 'fold': pindex + 1, 'folds': len(people)}), flush=True)
    result = {'seed': seed, 'controls': controls, 'frame_mapping': 'centered', 'population': {'tokens': count, 'speakers': len(people), 'classes': len(classes), 'words': len(set(words))}, 'outcomes': {'speaker': {'macro': float(weights @ baseline[:, 0]), 'micro': float(baseline[:, 0].mean())}, 'word': {'macro': float(weights @ baseline[:, 1]), 'micro': float(baseline[:, 1].mean())}}, 'context_strata': [], 'randomization_freedom': {key: dict(value) for key, value in freedom.items()}, 'forced_prediction_audit': forced}
    for name in names:
        replicate_scores = weights @ draws[name]
        result['outcomes'][name] = {'macro': float(replicate_scores.mean()), 'micro': float(draws[name].mean()), 'monte_carlo_se': float(replicate_scores.std(ddof=1) / np.sqrt(controls))}
    for present in (False, True):
        mask = overlap == present
        row = {'context_available': present, 'tokens': int(mask.sum()), 'classes': len(set(y[mask])), 'words': len(set(words[mask])), 'speakers': len(set(speakers[mask])), 'speaker_class_cells': len(set(zip(speakers[mask], y[mask]))), 'micro': {'speaker': float(baseline[mask, 0].mean()), 'word': float(baseline[mask, 1].mean())}}
        for name in names:
            row['micro'][name] = float(draws[name][mask].mean())
        result['context_strata'].append(row)
    gap = result['outcomes']['class_context']['macro'] - result['outcomes']['word']['macro']
    forced['all_cell_macro_gap'] = gap
    forced['derived_nonforced_macro_gap'] = gap / (1 - forced['macro_weight'])
    return result

def evaluate_sensitivity(corpus, x, controls=25, seed=20260928):
    require(controls >= 1, 'E_E_CONTROLS')
    s, w, y = (corpus.speakers, corpus.words, corpus.y)
    classes = corpus.classes
    people = sorted(set(s))
    rng = np.random.default_rng(seed)
    totals = np.zeros((len(people), len(classes)))
    correct = np.zeros((len(people), len(classes), 3))
    minimum = len(y)
    cells = 0
    for si, speaker in enumerate(people):
        train, test = (s != speaker, s == speaker)
        by_class = [np.flatnonzero(train & (y == ci)) for ci in range(len(classes))]
        base = np.stack([x[ix].mean(axis=0) for ix in by_class])
        for word in sorted(set(w[test])):
            test_ix = np.flatnonzero(test & (w == word))
            keep = [ix[w[ix] != word] for ix in by_class]
            require(all((len(ix) > 0 for ix in keep)), 'E_E_SENSITIVITY_EMPTY_CLASS')
            require(all((np.all(w[ix] != word) and np.all(s[ix] != speaker) for ix in keep)), 'E_E_SENSITIVITY_EXCLUSION')
            minimum = min(minimum, min(map(len, keep)))
            held = np.stack([x[ix].mean(axis=0) for ix in keep])
            pred_base = cdist(x[test_ix], base).argmin(axis=1)
            pred_held = cdist(x[test_ix], held).argmin(axis=1)
            control = np.zeros(len(test_ix))
            for replicate in range(controls):
                centers = np.stack([x[rng.choice(ix, size=len(kk), replace=False)].mean(axis=0) for ix, kk in zip(by_class, keep)])
                control += (cdist(x[test_ix], centers).argmin(axis=1) == y[test_ix]) / controls
            for ci in range(len(classes)):
                mask = y[test_ix] == ci
                totals[si, ci] += mask.sum()
                correct[si, ci, 0] += np.sum(pred_base[mask] == ci)
                correct[si, ci, 1] += np.sum(pred_held[mask] == ci)
                correct[si, ci, 2] += control[mask].sum()
            cells += 1
    require(np.all(totals > 0), 'E_E_SENSITIVITY_MACRO_CELL')
    accuracy = correct / totals[:, :, None]
    names = ('speaker', 'word', 'class')
    return {'seed': seed, 'controls': controls, 'macro': dict(zip(names, accuracy.mean(axis=(0, 1)).tolist())), 'micro': dict(zip(names, (correct.sum(axis=(0, 1)) / totals.sum()).tolist())), 'cells': cells, 'minimum_training_tokens': minimum}

def plot_comparison(comparison, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    SCHEMES = ('class', 'class_left', 'class_right', 'class_both', 'class_both_speaker')
    LABELS = ('Classe', 'Classe + anterior', 'Classe + seguinte', 'Classe + ambos', 'Classe + ambos + falante')
    rows = [comparison['partitions'][s] for s in SCHEMES]
    forced = np.array([r['forced_macro_weight'] for r in rows]) * 100
    variable_zero = np.array([r['uniform_macro_weighted_variable_zero_probability'] for r in rows]) * 100
    variable_positive = np.array([r['uniform_macro_weighted_any_associated_segment_probability'] for r in rows]) * 100
    expectation = np.array([r['expected_held_retention_fraction'] for r in rows]) * 100
    maximum = np.array([r['sharp_maximum_held_retention_fraction'] for r in rows]) * 100
    if not np.allclose(forced + variable_zero + variable_positive, 100, atol=1e-12):
        raise ValueError('E_FIGURE_WEIGHTS')
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'}):
        fig, axes = plt.subplots(1, 2, figsize=(16 / 2.54, 10.3 / 2.54), sharey=True)
        y = np.arange(len(rows))
        axes[0].barh(y, forced, color='0.25', label='Igualdade obrigatória', height=0.62)
        axes[0].barh(y, variable_zero, left=forced, color='white', edgecolor='0.3', hatch='////', linewidth=0.5, label='Variável: nenhum registro', height=0.62)
        axes[0].barh(y, variable_positive, left=forced + variable_zero, color='0.72', edgecolor='0.3', linewidth=0.5, label='Variável: ≥ 1 registro', height=0.62)
        axes[0].set_title('A. Composição\nda comparação', loc='left', pad=12)
        axes[0].set_xlabel('Peso macro esperado\ndo teste (%)')
        axes[0].set_xlim(0, 100)
        axes[0].set_xticks([0, 50, 100])
        axes[0].set_yticks(y, LABELS)
        axes[0].invert_yaxis()
        for i, (e, u) in enumerate(zip(expectation, maximum)):
            axes[1].plot([e, u], [i, i], color='0.5', linewidth=1.5)
        axes[1].scatter(expectation, y, marker='o', facecolor='black', s=28, label='Esperança')
        axes[1].scatter(maximum, y, marker='D', facecolor='white', edgecolor='black', s=27, label='Máximo exato')
        axes[1].set_title('B. Densidade\ndos registros', loc='left', pad=12)
        axes[1].set_xlabel('Incidências associadas\nretidas (%)')
        axes[1].set_xlim(-3, 103)
        axes[1].set_xticks([0, 50, 100])
        for axis in axes:
            axis.spines[['top', 'right', 'left']].set_visible(False)
            axis.tick_params(axis='y', length=0)
            axis.set_axisbelow(True)
            axis.grid(axis='x', color='0.90', linewidth=0.6)
        axes[0].legend(loc='upper left', bbox_to_anchor=(-0.05, -0.34), frameon=False, handlelength=1.2, fontsize=10, borderaxespad=0)
        axes[1].legend(loc='upper left', bbox_to_anchor=(-0.05, -0.34), frameon=False, handlelength=1.2, fontsize=10, borderaxespad=0)
        fig.subplots_adjust(left=0.33, right=0.97, top=0.82, bottom=0.36, wspace=0.4)
        out.mkdir(parents=True, exist_ok=True)
        for extension in ('pdf', 'png', 'jpg', 'svg'):
            extra = {'metadata': {'Title': None, 'Author': None, 'Subject': None, 'Keywords': None, 'Creator': None, 'Producer': None, 'CreationDate': None, 'ModDate': None}} if extension == 'pdf' else {'metadata': {'Software': None}} if extension == 'png' else {}
            fig.savefig(out / ('comparison_support.' + extension), dpi=350, facecolor='white', **extra)
        plt.close(fig)
    svg = out / 'comparison_support.svg'
    svg.write_text(re.sub('<metadata>.*?</metadata>', '', svg.read_text(encoding='utf-8'), flags=re.S), encoding='utf-8')

def extract_features(corpus, audio_directory, output):
    import librosa
    import soundfile
    audio_directory = Path(audio_directory).resolve()
    require(audio_directory.is_dir(), "E_AUDIO_DIRECTORY")
    paths = [audio_directory / (row["filename"] + ".wav") for row in corpus.metadata]
    missing = [path.name for path in paths if not path.is_file()]
    require(not missing, "E_MISSING_AUDIO:" + ",".join(missing))
    destination = output / "features"
    require(destination.resolve() != (corpus.root / "data/features").resolve(), "E_PROTECTED_FEATURES")
    comparisons = []
    for row, path in zip(corpus.metadata, paths):
        with wave.open(str(path), "rb") as stream:
            require((stream.getframerate(), stream.getnchannels(), stream.getnframes(), stream.getsampwidth()) ==
                    (row["sample_rate"], row["channels"], row["samples"], row["sample_width"]), "E_WAV_HEADER:" + path.name)
            digest = hashlib.sha256(stream.readframes(stream.getnframes())).hexdigest()
        require(digest == row["pcm_sha256"], "E_WAV_PCM:" + path.name)
        signal, rate = librosa.load(str(path), sr=None, mono=True)
        emphasized = np.empty_like(signal)
        emphasized[0] = signal[0]
        emphasized[1:] = signal[1:] - .97 * signal[:-1]
        fft = 1 << (round(.025 * rate) - 1).bit_length()
        hop = round(.01 * rate)
        static = librosa.feature.mfcc(y=emphasized, sr=rate, n_mfcc=13, n_fft=fft,
                                      hop_length=hop, window="hann", center=True, n_mels=26, htk=False)
        raw = np.concatenate([static, librosa.feature.delta(static, order=1),
                              librosa.feature.delta(static, order=2)], axis=0).T.astype(np.float32)
        normalized = (raw - raw.mean(axis=0, keepdims=True)) / (raw.std(axis=0, keepdims=True) + 1e-8)
        features = np.stack([raw, normalized], axis=-1)
        require(features.shape == (1 + row["samples"] // 441, 39, 2) and np.isfinite(features).all(), "E_EXTRACTED_FEATURES")
        reference_path = corpus.root / "data/features" / (row["filename"] + ".npz")
        with np.load(reference_path, allow_pickle=False) as zipped:
            reference = zipped["features"]
        difference = float(np.abs(features - reference).max())
        save_features(destination / (row["filename"] + ".npz"), features)
        comparisons.append({"filename": row["filename"], "frames": len(features), "maximum_absolute_difference": difference})
        print(json.dumps({"stage": "extract", "recording": len(comparisons), "recordings": len(paths)}), flush=True)
    report = {"status": "PASS_EXTRACTION", "recordings": comparisons, "pcm_hashes": "PASS",
              "numpy": np.__version__, "librosa": librosa.__version__, "soundfile": soundfile.__version__}
    write_json(output / "extraction.json", report)
    return report


def check_sources(root=ROOT):
    source = (root / "analysis.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    require(not any(token.type == tokenize.COMMENT for token in tokenize.generate_tokens(io.StringIO(source).readline)), "E_PYTHON_COMMENTS")
    require(not any(ast.get_docstring(node) for node in ast.walk(tree) if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))), "E_PYTHON_DOCSTRINGS")
    lean = (root / "Proofs.lean").read_text(encoding="utf-8")
    require("/-" not in lean and "--" not in lean, "E_LEAN_COMMENTS")
    require(not re.search(r"\b(sorry|admit|sorryAx|axiom|native_decide)\b", lean), "E_LEAN_PROOF_CONSTRUCT")
    roots = set(re.findall(r"^#print axioms (\S+)$", lean, re.M))
    require(len(roots) == 17, "E_LEAN_ROOTS")
    return {"python_comments": 0, "python_docstrings": 0, "lean_comments": 0, "lean_roots": 17,
            "lean_compilation": "NOT_RUN", "python_sha256": sha(root / "analysis.py"), "lean_sha256": sha(root / "Proofs.lean")}


def check_manifest(root=ROOT):
    path = root / "SHA256SUMS"
    require(path.is_file(), "E_MANIFEST_MISSING")
    checked = []
    for line in path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        member = Path(relative)
        require(not member.is_absolute() and ".." not in member.parts, "E_MANIFEST_PATH")
        target = root / member
        require(target.is_file() and not target.is_symlink(), "E_MANIFEST_MEMBER:" + relative)
        require(sha(target) == expected, "E_MANIFEST_HASH:" + relative)
        checked.append(relative)
    require(len(checked) == len(set(checked)), "E_MANIFEST_DUPLICATE")
    require("analysis.py" in checked and "Proofs.lean" in checked, "E_MANIFEST_SOURCES")
    return {"status": "PASS", "files": len(checked)}


def check_representations():
    metadata = [{"filename": f"synthetic{i:02d}"} for i in range(37)]
    rows, arrays = [], []
    for speaker in range(37):
        array = np.zeros((20, 39, 2), dtype=np.float32)
        array[:, :, 0] = 1000000 + speaker
        for k, (label, word, value) in enumerate((("a", " ALPHA ", 1), ("a", "beta", 1),
                                                 ("b", "Alpha", -1), ("b", " BETA ", -1), ("c", "singleton", 5))):
            array[4 * k:4 * k + 4, :, 1] = value + speaker / 10000
            rows.append({"speaker": speaker, "word": word.strip().casefold(), "label": label,
                         "left": "x", "right": "y", "start": 4 * k / 100,
                         "end": (4 * k + 4) / 100, "archived_first": 4 * k,
                         "archived_last": 4 * k + 3, "segment_index": k})
        arrays.append(array)
    class_words = defaultdict(set)
    for row in rows:
        class_words[row["label"]].add(row["word"])
    classes = sorted(label for label, words in class_words.items() if len(words) >= 2)
    valid = np.array([row["label"] in classes for row in rows])
    selected = [row for row, keep in zip(rows, valid) if keep]
    speakers = np.array([row["speaker"] for row in selected])
    words = np.array([row["word"] for row in selected])
    labels = np.array([row["label"] for row in selected])
    contexts = np.array([json.dumps(("x", "y"))] * len(selected))
    y = np.searchsorted(classes, labels)
    records = [(row["speaker"], row["word"], row["label"], "x", "y") for row in selected]
    corpus = Corpus(ROOT, metadata, rows, valid, classes, speakers, words, labels, contexts,
                    y, records, np.full(len(selected), 1 / len(selected)), {}, 0)
    x = represent(corpus, arrays)
    require(x.shape == (148, 39) and classes == ["a", "b"], "E_SYNTHETIC_ELIGIBILITY")
    require(set(words) == {"alpha", "beta"}, "E_SYNTHETIC_WORDS")
    require(np.allclose(x[0], 1) and np.allclose(x[2], -1), "E_SYNTHETIC_CHANNEL")
    require(np.array_equal(x, represent(corpus, arrays, mapping="archived")), "E_SYNTHETIC_MAPPING")
    require(np.array_equal(x[:, :13], represent(corpus, arrays, block="static")), "E_SYNTHETIC_STATIC")
    require(central_indices(np.arange(1), .4).tolist() == [0], "E_SYNTHETIC_SHORT_INTERVAL")
    result = evaluate_sensitivity(corpus, x, controls=2)
    require(all(abs(score - 1) < 1e-12 for score in result["macro"].values()), "E_SYNTHETIC_CLASSIFICATION")
    require(result["cells"] == 74 and result["minimum_training_tokens"] == 36, "E_SYNTHETIC_TRAINING")
    return {"status": "PASS", "tokens": 148, "classes": 2, "cells": 74,
            "channel": "cmvn", "word_normalization": "PASS", "frame_mapping": "PASS", "accuracy": 1.}


def compare_targets(actual, expected, path="", tolerance=1e-10):
    if isinstance(expected, dict):
        for key, value in expected.items():
            require(key in actual, "E_TARGET_KEY:" + path + "." + key)
            compare_targets(actual[key], value, path + "." + key, tolerance)
    elif isinstance(expected, list):
        require(len(actual) == len(expected), "E_TARGET_LENGTH:" + path)
        for index, (value, target) in enumerate(zip(actual, expected)):
            compare_targets(value, target, path + "." + str(index), tolerance)
    elif isinstance(expected, float):
        require(math.isfinite(actual) and abs(actual - expected) <= tolerance, "E_TARGET_VALUE:" + path)
    else:
        require(actual == expected, "E_TARGET_VALUE:" + path)


def verify_structure(corpus, comparison, structure, timing, targets):
    compare_targets(corpus.population, targets["population"], "population")
    compare_targets(comparison, targets["comparison"], "comparison")
    compare_targets(structure, targets["structure"], "structure")
    compare_targets(timing["counts_before_eligibility"], targets["timing"], "timing")
    require(timing["excluded_without_archived_frames"] == 36, "E_PREARCHIVE_EXCLUSIONS")
    rows = targets["table2"]
    for name, expected in rows.items():
        row = comparison["partitions"][name]
        actual = [row["forced_cells"], round(row["forced_macro_weight"] * 100, 2),
                  round(row["expected_held_retention_fraction"] * 100, 2),
                  round(row["sharp_maximum_held_retention_fraction"] * 100, 2)]
        require(actual == expected, "E_TABLE2:" + name)
    maximum = max(timing["delta_formula_max_absolute_residual"])
    require(maximum < 1e-4, "E_REEXTRACTED_DERIVATIVE_RESIDUAL")
    return {"status": "PASS_COUNTS_AND_TIMING", "derivative_max_absolute_residual": maximum,
            "derivative_matches_manuscript_2e_6_bound": maximum <= 2e-6,
            "reextracted_derivative_tolerance": 1e-4}


def verify_formal(root, output):
    check_sources(root)
    require(shutil.which("lean") is not None and shutil.which("lake") is not None, "E_LEAN_TOOLCHAIN_UNAVAILABLE")
    roots = set(re.findall(r"^#print axioms (\S+)$", (root / "Proofs.lean").read_text(encoding="utf-8"), re.M))
    allowed = {"propext", "Classical.choice", "Quot.sound"}
    commands = []
    with tempfile.TemporaryDirectory(prefix="proofs_") as directory:
        clean = Path(directory)
        for name in ("Proofs.lean", "lean-toolchain", "lakefile.toml"):
            shutil.copyfile(root / name, clean / name)
        def run(name, command):
            result = subprocess.run(command, cwd=clean, text=True, capture_output=True)
            text = result.stdout + result.stderr
            (output / (name + ".log")).write_text(text, encoding="utf-8")
            commands.append({"check": name, "returncode": result.returncode})
            require(result.returncode == 0, "E_FORMAL:" + name)
            return text
        version = run("lean_version", ["lean", "--version"])
        require("version 4.34.0," in version, "E_LEAN_VERSION")
        def axioms(text):
            found = {name: {value.strip() for value in values.split(",") if value.strip()}
                     for name, values in re.findall(r"'([^']+)' depends on axioms: \[([^\]]*)\]", text)}
            for name in re.findall(r"'([^']+)' does not depend on any axioms", text):
                found[name] = set()
            require(set(found) == roots, "E_AXIOM_ROOTS")
            require(all(values <= allowed for values in found.values()), "E_AXIOM_DEPENDENCY")
            return {name: sorted(values) for name, values in sorted(found.items())}
        built = axioms(run("lean_build", ["lake", "build"]))
        replayed = axioms(run("lean_trust0", ["lake", "env", "lean", "--trust=0", "Proofs.lean"]))
        require(built == replayed, "E_AXIOM_REPLAY")
    report = {"status": "PASS_CLEAN_BUILD_AND_TRUST0", "roots": len(roots), "axioms": built,
              "version": version.strip(), "commands": commands, "source_sha256": sha(root / "Proofs.lean")}
    write_json(output / "formal.json", report)
    return report


def export_tables(output, comparison, primary=None, sensitivities=None):
    def save(name, header, rows):
        with (output / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)
    row = comparison["partitions"]["class_both"]
    forced = row["forced_cells"]
    cells = row["cells"]
    tokens = comparison["population"]["tokens"]
    p = row["nonforced_any_probability_distribution"]["mean"]
    save("table1.csv", ["condition", "cells", "test_segments", "macro_weight_percent", "mean_Q_percent"],
         [["forced", forced, row["forced_test_tokens"], 100 * row["forced_macro_weight"], 0],
          ["variable", cells - forced, tokens - row["forced_test_tokens"], 100 * row["possible_macro_weight"], 100 * p],
          ["all", cells, tokens, 100, 100 * p * (cells - forced) / cells]])
    save("table2.csv", ["partition", "forced_cells", "forced_macro_weight_percent", "expected_retention_percent", "maximum_retention_percent"],
         [[name, value["forced_cells"], 100 * value["forced_macro_weight"], 100 * value["expected_held_retention_fraction"],
           100 * value["sharp_maximum_held_retention_fraction"]] for name, value in comparison["partitions"].items()])
    if primary is not None:
        save("table3.csv", ["condition", "macro_percent", "micro_percent", "monte_carlo_se_pp"],
             [[name, 100 * value["macro"], 100 * value["micro"],
               100 * value["monte_carlo_se"] if "monte_carlo_se" in value else ""] for name, value in primary["outcomes"].items()])
    if primary is not None and sensitivities is not None:
        main = primary["outcomes"]
        values = [["centered60", 100, 100 * main["speaker"]["macro"], 100 * main["word"]["macro"],
                   100 * main["class"]["macro"], 100 * (main["class"]["macro"] - main["word"]["macro"])]]
        for name in ("centered40", "centered100", "centered_static", "archived60"):
            row = sensitivities[name]
            scores = row["macro"]
            values.append([name, row["controls"], 100 * scores["speaker"], 100 * scores["word"], 100 * scores["class"],
                           100 * (scores["class"] - scores["word"])])
        save("table4.csv", ["representation", "controls", "speaker_percent", "word_percent", "class_percent", "class_minus_word_pp"], values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("verify", "support", "acoustic", "all", "extract", "formal", "plot"), nargs="?", default="verify")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--audio-dir", type=Path, default=ROOT / "audio")
    parser.add_argument("--features", type=Path)
    args = parser.parse_args()
    require(not sys.flags.optimize, "E_PYTHON_OPTIMIZATION")
    output = args.output.resolve()
    require(output != ROOT and not output.is_relative_to(ROOT / "data") and not output.is_relative_to(ROOT / "reference") and not output.is_relative_to(ROOT / "audio"), "E_PROTECTED_OUTPUT")
    require(not args.output.is_symlink(), "E_OUTPUT_SYMLINK")
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    if args.command == "formal":
        result = verify_formal(ROOT, output)
        print(json.dumps({"status": result["status"], "roots": result["roots"]}))
        return
    manifest = check_manifest()
    sources = check_sources()
    corpus = load_corpus()
    targets = json.loads((ROOT / "reference/targets.json").read_text(encoding="utf-8"))
    if args.command == "extract":
        result = extract_features(corpus, args.audio_dir, output)
        print(json.dumps({"status": result["status"], "recordings": len(result["recordings"])}))
        return
    comparison = analyze(corpus.records)
    write_json(output / "comparison.json", comparison)
    if args.command == "plot":
        plot_comparison(comparison, output)
        export_tables(output, comparison)
        print(json.dumps({"status": "PASS_FIGURE"}))
        return
    report = {"manifest": manifest, "sources": sources, "command": args.command, "population": corpus.population,
              "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
              "formal": "NOT_RUN", "acoustic": "NOT_RUN", "sensitivities": "NOT_RUN"}
    report["synthetic"] = synthetic_checks()
    report["representations"] = check_representations()
    support, degeneracy = audit_support(corpus.speakers, corpus.words, corpus.labels, corpus.contexts, corpus.classes)
    structure = {"support": support, "degeneracy": degeneracy}
    write_json(output / "structure.json", structure)
    arrays = load_arrays(corpus, args.features)
    timing = audit_timing(corpus, arrays)
    write_json(output / "timing.json", timing)
    report["structure"] = verify_structure(corpus, comparison, structure, timing, targets)
    primary = None
    sensitivities = None
    if args.command in ("acoustic", "all"):
        primary = evaluate_primary(corpus, represent(corpus, arrays))
        write_json(output / "primary.json", primary)
        compare_targets(primary, targets["primary"], "primary")
        report["acoustic"] = "PASS_REGRESSION"
    if args.command == "all":
        sensitivities = {}
        for name, controls, fraction, block, mapping in (
            ("centered60", 100, .6, "all", "centered"),
            ("centered40", 25, .4, "all", "centered"),
            ("centered100", 25, 1., "all", "centered"),
            ("centered_static", 25, .6, "static", "centered"),
            ("centered60_25", 25, .6, "all", "centered"),
            ("archived60", 100, .6, "all", "archived")):
            value = evaluate_sensitivity(corpus, represent(corpus, arrays, fraction, block, mapping), controls)
            value.update(fraction=fraction, block=block, mapping=mapping)
            sensitivities[name] = value
            write_json(output / "sensitivities.json", sensitivities)
            compare_targets(value, targets["sensitivities"][name], "sensitivities." + name)
            print(json.dumps({"stage": "sensitivity", "name": name, "status": "PASS"}), flush=True)
        report["sensitivities"] = "PASS_REGRESSION"
        plot_comparison(comparison, output)
    export_tables(output, comparison, primary, sensitivities)
    report["manifest_after"] = check_manifest()
    require(report["manifest_after"] == manifest, "E_INPUT_CHANGED")
    report["status"] = "PASS_EXECUTED_CHECKS"
    write_json(output / "verification.json", report)
    print(json.dumps({"status": report["status"], "acoustic": report["acoustic"], "formal": report["formal"]}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"status": "FAIL", "error": str(error), "type": type(error).__name__}), file=sys.stderr)
        sys.exit(1)
