# Science Corpus Enrichment Plan

This plan extends Wolfe from chunk search into a research knowledge substrate for
the `science` Library corpus. The goal is to preserve document provenance,
derive bibliographic and conceptual metadata, expose useful search facets, and
feed higher-level GroundRecall, CiteGeist, doclift, and Didactopus workflows.

## Principles

- Keep every derived assertion tied to a source file, extraction method, byte or
  page span when available, and model/tool version.
- Store raw extracted candidates separately from verified or normalized records.
- Prefer deterministic extractors first, then LLM passes over bounded text spans.
- Treat bibliographic metadata as uncertain until cross-checked by DOI, ISBN,
  Crossref, OpenLibrary, WorldCat, CiteGeist, or local citation evidence.
- Keep public-site search needs separate from privileged research analysis.

## Per-Source Metadata

Create one metadata record per ingested source. A sidecar JSON file is useful for
inspection and portability, but the authoritative records should also be loaded
into a searchable database.

Suggested sidecar path:

```text
<metadata-root>/<source-sha256-prefix>/<source-filename>.wolfe-meta.json
```

Core fields:

- `source_id`: stable hash over normalized path plus file hash.
- `path`, `file_name`, `extension`, `parent_dir`.
- `size_bytes`, `mtime`, `sha256`, optional fast hash for change detection.
- `ingested_at`, `wolfe_version`, extractor versions.
- `page_count`, `word_count`, `text_chunk_count`, `image_chunk_count`.
- `language`, OCR confidence if available, text extraction quality flags.
- `document_type`: book, article, chapter, report, note, message, web capture,
  slide deck, dataset, archive, unknown.
- `rights_scope`: owned, foundation-owned, licensed, third-party, unknown.
- `canonical_title`, `subtitle`, `authors`, `editors`, `publisher`, `journal`,
  `volume`, `issue`, `pages`, `year`, `date`, `doi`, `isbn`, `url`.
- `bibtex_candidate`, `bibtex_confidence`, `bibtex_sources`.
- `collections` and local taxonomy tags derived from path and content.

For search display, expose concise fields directly in the Wolfe index:

- title or best filename fallback.
- author/date when known.
- document type.
- page count and file size.
- confidence badges for OCR/text extraction and bibliography match.
- topic/concept facets.

## Pass 1: Bibliographic Identification

Inputs:

- file path and filename.
- first pages or first N text chunks.
- PDF metadata, document properties, OCR title pages.
- reference-list self-citations.

Outputs:

- one or more `bibliographic_candidate` records.
- normalized BibTeX when confidence is high enough.
- unresolved candidate with explicit reason when not high confidence.

Implementation steps:

1. Extract deterministic metadata with `pdfinfo`, `exiftool`, LibreOffice
   properties, archive manifests, and filename/path heuristics.
2. Use CiteGeist to resolve title/author/year/DOI candidates.
3. Query Crossref/OpenAlex/ISBN/OpenLibrary where appropriate.
4. Store all candidates with scores and evidence spans.
5. Promote a `canonical_bibliographic_record` only when confidence passes a
   threshold or after human review.

## Pass 2: References And Citations

Extract formal and informal references from every source.

Target record types:

- `reference_entry`: bibliography/reference-list item.
- `inline_citation`: parenthetical, footnote, numbered citation, author-year.
- `casual_reference`: less formal source indication such as "Darwin wrote...",
  "as Gould argued...", "the Supreme Court decision...", "the ICR article...".
- `self_reference`: source referring to its own figures, chapters, or prior
  editions.

Fields:

- `source_id`, `chunk_id`, page/offset span.
- original text span.
- normalized citation string.
- parsed authors, year, title, venue, locator.
- candidate DOI/ISBN/URL.
- linked bibliographic record if resolved.
- extraction method and confidence.

Useful features:

- Reference graph: source cites source, source discusses source, source quotes
  source.
- Missing-citation detector: claims with named works or authors but no formal
  reference.
- Bibliography quality report: duplicate references, unresolved references,
  broken DOI/URL, inconsistent author/year/title forms.

## Pass 3: Concept Phrases

Extract concepts useful for research, site search, and argument mapping.

Target record:

- `concept_phrase`: exact phrase and normalized phrase.
- `source_id`, `chunk_id`, page/offset span.
- `role`: topic, term, organism, person, place, institution, method, claim,
  argument, analogy, law/case, dataset, model, geological period, taxon.
- `definition_type`: explicit definition, implicit description, contrast,
  connotation, denotation, example, negation, disputed usage.
- `definition_text` or short extracted context when present.
- `polarity_or_stance`: asserted, denied, criticized, attributed, quoted,
  hypothetical, historical.
- `confidence`, extraction method, model/tool version.

Important detail:

Connotation and denotation should remain attached to the usage instance, not
only to the normalized concept. Creation/evolution discourse often uses the same
phrase with different intended framing, so search should be able to distinguish
neutral scientific usage from rhetorical or antievolutionary usage.

## Pass 4: Quotations

Extract quoted material and preserve provenance.

Target record:

- `quote_text`, normalized quote text, and exact source text span.
- `source_id`, file name, page/offset span, surrounding chunk IDs.
- quote introducer and attribution text.
- quoted entity/person/work when stated.
- associated citation/reference entry if present.
- quote type: direct quotation, block quote, epigraph, scare quote, dialogue,
  statute/case excerpt, embedded quotation.
- ownership context and quotation-length risk flags.

Useful features:

- Quote verification against cited source when the cited source is also in the
  corpus or externally resolvable.
- Misquote/paraphrase detector.
- Quote reuse graph across sources and web sites.
- Short-excerpt policy support for third-party materials.

## Pass 5: Claims And Argument Structure

For grounding web sites and research, the system should identify claims and how
sources support, dispute, or contextualize them.

Target record:

- claim text and normalized claim.
- source span.
- claim type: factual, historical, bibliographic, definitional, causal,
  methodological, theological/philosophical, legal/policy, rhetorical.
- stance: asserts, denies, questions, qualifies, quotes, attributes.
- evidence links: references, quotes, concepts, figures, tables.
- checkability: directly checkable, needs specialist review, opinion/argument.

This is where GroundRecall/Didactopus should eventually receive a graph-ready
form: source -> span -> claim -> evidence -> citation -> concept.

## Pass 6: Figures, Tables, And Captions

Many science sources carry evidence in figures and tables.

Extract:

- figure/table captions and page locations.
- table text where reliably extractable.
- image descriptions for diagrams and scanned figures.
- cross-references from text to figures/tables.
- cited data sources for tables and graphs.

These should be searchable and linked back to source pages.

## Pass 7: Named Entities And Authorities

Extract people, organizations, journals, publishers, courts, agencies, projects,
and named datasets.

Useful outputs:

- authority records with aliases.
- entity co-occurrence graphs.
- author/institution bibliographies.
- creation/evolution organization and publication provenance.

## Pass 8: Corpus Quality And Review Queues

Generate review queues rather than pretending all extraction is equally sound.

Queues:

- low OCR/text quality.
- no bibliographic candidate.
- conflicting bibliographic candidates.
- references unresolved by CiteGeist.
- high-value quotes missing cited source.
- third-party material with long extracted quotation spans.
- documents likely duplicated under different names.
- sources with unclear rights scope.

## Storage Layout

Use multiple stores with clear responsibility:

- LanceDB: chunk embeddings, source-level embeddings, concept embeddings, quote
  embeddings.
- SQLite or PostgreSQL: normalized metadata, bibliographic records, references,
  concepts, quotes, claims, review queues.
- Sidecar JSON: portable per-source derived metadata and extraction evidence.
- BibTeX export: canonical and candidate bibliographic records.
- GroundRecall/doclift graph export: provenance-preserving claim/concept/citation
  relationships.

Minimum tables:

- `sources`
- `source_files`
- `source_chunks`
- `bibliographic_candidates`
- `bibliographic_records`
- `references`
- `concept_mentions`
- `quotes`
- `claims`
- `figures_tables`
- `entities`
- `source_relations`
- `review_items`
- `extraction_runs`

## Search And UI Features

Add search modes beyond semantic chunk search:

- source search: title, author, year, path, document type, collection.
- bibliography search: BibTeX, DOI, ISBN, author, cited title.
- reference search: documents citing an author/work/topic.
- concept search: normalized concept, usage role, definition type, stance.
- quote search: exact, fuzzy, attribution, cited source.
- claim search: stance and evidence availability.
- provenance view: every result shows source metadata, chunk/page locator,
  extraction confidence, and linked citations.

For public search, expose only source and chunk display appropriate to the site.
For privileged research, expose claims, quote-risk flags, unresolved citations,
and review queues.

## Initial Implementation Order

1. Build a source metadata sidecar generator over the existing `science` index.
2. Add a metadata database with `sources`, `extraction_runs`, and
   `bibliographic_candidates`.
3. Integrate CiteGeist bibliographic resolution and BibTeX candidate storage.
4. Surface source metadata in Wolfe web search results.
5. Add reference extraction, first deterministic regex/parser pass, then LLM
   cleanup and resolution.
6. Add concept and quote extraction passes over bounded chunks.
7. Add review queues and reports.
8. Export graph-ready records for GroundRecall/doclift/Didactopus.

## Additional Features Worth Adding

- Duplicate and near-duplicate detection across files and editions.
- Edition/part grouping for split PDFs from the same book.
- Archive expansion provenance, so records from `sor_ocr.zip` retain archive
  membership.
- OCR confidence and "needs OCR" routing.
- Citation freshness checks for URLs, DOI links, and Archive.org snapshots.
- Topic timeline generation from dated sources.
- Source reliability annotations separated from extracted facts.
- Human correction layer for canonical metadata and concept normalization.
- Stable citation handles for web-site authors to cite internal corpus evidence.
- Export packs for site builds: BibTeX, JSON metadata, quote snippets,
  source cards, and topic bibliographies.
