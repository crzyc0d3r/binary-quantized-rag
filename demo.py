"""Deterministic mechanics check using explicit fixture vectors, without networks."""
import json
from rag import Chunk, MemoryIndex, evidence_prompt


def run():
    chunks = [Chunk("Packed binary vectors use one bit per dimension.", "packing.md"),
              Chunk("Hamming distance counts the differing binary dimensions.", "distance.md"),
              Chunk("A generator must cite its retrieved evidence.", "generation.md")]
    # These hand-written vectors test storage/search mechanics, not language understanding.
    vectors = [[1] * 8, [-1] * 8, [1, -1] * 4]
    index = MemoryIndex(chunks, vectors)
    hits = index.search([1] * 8, top_k=2)
    result = dict(mode="fixture_vectors_no_model", quality_evidence=False,
                  float32_payload_bytes=3 * 8 * 4, packed_payload_bytes=index.bits.nbytes,
                  payload_reduction=32, hits=hits,
                  prompt=evidence_prompt("How are vectors packed?", hits))
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
