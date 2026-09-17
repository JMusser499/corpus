"""Status inspects source evidence without loading extraction models or writing."""
import json

import pytest

from pipeline import taxonomy_ingest
from pipeline.build_inputs import source_input_drift
from pipeline.stages import _load_pipeline_state
from tests import test_metadata_resume

corpus = test_metadata_resume.corpus


def configure(corpus, extra=""):
    corpus.config.write_text("input_pdfs: library\nbib: library.bib\ngrobid:\n  disable: true\n" + extra)


def test_current_and_changed_bibliography_status_is_read_only(corpus):
    configure(corpus)
    corpus.run()
    assert source_input_drift(corpus.output, corpus.config)["differences"] == {}
    corpus.set_bib(first="Revised")
    before = {p: p.read_bytes() for p in corpus.output.rglob("*") if p.is_file()}
    drift = source_input_drift(corpus.output, corpus.config)
    assert drift["documents_with_differences"] == 1
    assert drift["differences"][corpus.hd().name] == {"metadata_extraction": ["bib_entry_sha256"]}
    assert {p: p.read_bytes() for p in corpus.output.rglob("*") if p.is_file()} == before
    corpus.run()
    assert not source_input_drift(corpus.output, corpus.config)["differences"]


def test_inventory_separates_rename_addition_and_removal(corpus):
    configure(corpus)
    corpus.run()
    sha = corpus.hd().name
    removed = corpus.hd("Second2002.pdf").name
    (corpus.source / "First2001.pdf").rename(corpus.source / "Renamed.pdf")
    (corpus.source / "Second2002.pdf").unlink()
    (corpus.source / "New.pdf").write_bytes(b"new source bytes")
    drift = source_input_drift(corpus.output, corpus.config)
    assert len(drift["added"]) == 1
    assert drift["removed"] == [removed]
    assert "filename" in drift["differences"][sha]["metadata_extraction"]
    assert drift["differences"][sha]["source_inventory"] == ["source paths"]


def test_annotation_addition_and_removal_are_reported(corpus):
    configure(corpus)
    corpus.run()
    lexicon = corpus.config.parent / "lexicon.yaml"
    lexicon.write_text("anatomy:\n  tentacle: {}\n")
    configure(corpus, "lexicon: lexicon.yaml\n")
    assert source_input_drift(corpus.output, corpus.config)["documents_with_differences"] == 2
    hd = corpus.hd()
    state = _load_pipeline_state(hd)
    state["stages"]["taxa_and_lexicon_extraction"] = dict(state["stages"]["text_chunking"])
    state["stages"]["taxa_and_lexicon_extraction"]["input_fingerprint"] = {"lexicons": {"anatomy": {"sha256": "old"}}}
    (hd / "pipeline_state.json").write_text(json.dumps(state))
    configure(corpus)
    changes = source_input_drift(corpus.output, corpus.config)["differences"][hd.name]
    assert changes["taxa_and_lexicon_extraction"] == ["lexicons.anatomy.sha256"]


def test_taxonomy_audit_uses_the_producers_stable_source_receipt(corpus, monkeypatch):
    configure(corpus)
    source = corpus.config.parent / "Taxon.tsv"
    source.write_text(
        "taxonID\tscientificName\tparentNameUsageID\n"
        "1\tAlpha\t\n"
        "2\tBeta\t1\n"
    )

    def ingest(*extra):
        monkeypatch.setattr(
            "sys.argv",
            ["ingest", str(corpus.output), "--source", "dwc", "--input", str(source), *extra],
        )
        assert taxonomy_ingest.main() == 0

    ingest()
    corpus.run(annotations=True)
    assert source_input_drift(corpus.output, corpus.config)["differences"] == {}

    # Rebuilding an unchanged source rewrites volatile SQLite timestamps but
    # preserves the receipt that identifies the consumed taxonomy input.
    before = (corpus.output / "taxonomy.sqlite").read_bytes()
    ingest("--rebuild")
    assert (corpus.output / "taxonomy.sqlite").read_bytes() != before
    assert source_input_drift(corpus.output, corpus.config)["differences"] == {}


def test_missing_configured_source_is_an_error_not_zero_drift(corpus):
    configure(corpus)
    corpus.run()
    corpus.bib.unlink()
    with pytest.raises(FileNotFoundError):
        source_input_drift(corpus.output, corpus.config)


def test_unconfigured_inventory_explicitly_reports_not_checked(corpus):
    corpus.run()
    assert source_input_drift(corpus.output, corpus.config)["available"] is False
