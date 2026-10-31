import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from demo import run
from rag import Chunk, MemoryIndex, MilvusIndex, NativePipeline, answer, evidence_prompt, load_chunks, pack_embeddings


class PackingTests(unittest.TestCase):
    def test_known_sign_pattern_and_zero(self):
        self.assertEqual(pack_embeddings([[1, -1, 0, 2, 0, -1, 3, 4]]).tolist(), [[147]])

    def test_payload_reduction(self):
        matrix = np.ones((7, 1024), dtype=np.float32)
        self.assertEqual(matrix.nbytes // pack_embeddings(matrix).nbytes, 32)

    def test_invalid_matrices(self):
        for value in [[], [1] * 8, [[1] * 7], [[]], [[np.nan] * 8], [[np.inf] * 8]]:
            with self.subTest(value=str(value)[:30]), self.assertRaises(ValueError):
                pack_embeddings(value)

    def test_hamming_and_stable_ties(self):
        index = MemoryIndex([Chunk(str(i), "fixture") for i in range(3)],
                            [[1] * 8, [-1] * 8, [1, -1] * 4])
        self.assertEqual([h["distance"] for h in index.search([1] * 8)], [0, 4, 8])
        self.assertEqual([h["id"] for h in index.search([0] * 8)], [1, 2, 0])

    def test_duplicate_distances_preserve_source_order(self):
        index = MemoryIndex([Chunk("a", "a"), Chunk("b", "b")], [[1] * 8] * 2)
        self.assertEqual([h["source"] for h in index.search([1] * 8)], ["a", "b"])

    def test_count_mismatch(self):
        with self.assertRaises(ValueError):
            MemoryIndex([Chunk("a", "a")], [[1] * 8] * 2)

    def test_empty_and_overlong_chunks(self):
        for chunks in [[], [Chunk(" ", "a")], [Chunk("a" * 65536, "a")], [Chunk("a", "b" * 2049)]]:
            with self.subTest(chunks=len(chunks)), self.assertRaises(ValueError):
                MemoryIndex(chunks, [[1] * 8])

    def test_dimension_mismatch(self):
        index = MemoryIndex([Chunk("a", "a")], [[1] * 8])
        with self.assertRaises(ValueError):
            index.search([1] * 16)

    def test_invalid_k(self):
        index = MemoryIndex([Chunk("a", "a")], [[1] * 8])
        for k in [0, -1, True, 1.5, "2"]:
            with self.subTest(k=k), self.assertRaises(ValueError):
                index.search([1] * 8, k)

    def test_demo_is_not_quality_evidence(self):
        result = run()
        self.assertFalse(result["quality_evidence"])
        self.assertEqual((result["float32_payload_bytes"], result["packed_payload_bytes"]), (96, 3))

    def test_evidence_and_empty_input(self):
        prompt = evidence_prompt("What?", [dict(source="a.md", text="A fact.")])
        self.assertIn("[1] a.md\nA fact.", prompt)
        for q, hits in [("", [dict(source="a", text="b")]), ("Q", [])]:
            with self.assertRaises(ValueError):
                evidence_prompt(q, hits)

    def test_answer_uses_native_interface(self):
        llm = MagicMock()
        llm.complete.return_value.text = "A fact. [1]"
        self.assertEqual(answer("What?", [dict(source="a", text="A fact.")], llm=llm), "A fact. [1]")
        self.assertIn("A fact.", llm.complete.call_args.args[0])


class NativeAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from pymilvus import MilvusClient
            from llama_index.core import SimpleDirectoryReader
        except ImportError as exc:
            raise unittest.SkipTest(f"Native SDKs unavailable: {exc}")

    def client(self):
        from pymilvus import MilvusClient
        client = MagicMock()
        client.has_collection.return_value = False
        client.create_schema.side_effect = MilvusClient.create_schema
        client.prepare_index_params.side_effect = MilvusClient.prepare_index_params
        return client

    def test_real_milvus_schema_and_packed_requests(self):
        client = self.client()
        index = MilvusIndex(client, [Chunk("a", "a.md"), Chunk("b", "b.md")],
                            [[1] * 8, [-1] * 8], collection="test_binary")
        created = client.create_collection.call_args.kwargs
        fields = {f.name: f for f in created["schema"].fields}
        from pymilvus import DataType
        self.assertEqual(fields["binary_vector"].dtype, DataType.BINARY_VECTOR)
        self.assertEqual(fields["binary_vector"].params["dim"], 8)
        self.assertIn("BIN_FLAT", str(created["index_params"]))
        self.assertIn("HAMMING", str(created["index_params"]))
        inserted = client.insert.call_args.kwargs["data"]
        self.assertEqual([row["binary_vector"] for row in inserted], [b"\xff", b"\x00"])
        client.search.return_value = [[dict(id=1, distance=0, entity=dict(context="b", source="b.md"))]]
        self.assertEqual(index.search([-1] * 8)[0]["source"], "b.md")
        self.assertEqual(client.search.call_args.kwargs["data"], [b"\x00"])

    def test_existing_collection_is_preserved(self):
        client = self.client()
        client.has_collection.return_value = True
        with self.assertRaises(ValueError):
            MilvusIndex(client, [Chunk("a", "a")], [[1] * 8], collection="existing")
        client.drop_collection.assert_not_called()
        client.create_collection.assert_not_called()

    def test_real_directory_reader_and_splitter(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "manual.txt").write_text("Binary vectors occupy one bit per dimension.", encoding="utf-8")
            chunks = load_chunks(directory)
        self.assertEqual(chunks, [Chunk("Binary vectors occupy one bit per dimension.", "manual.txt")])

    def test_native_groq_adapter_with_no_socket_http_transport(self):
        import httpx
        from llama_index.llms.groq import Groq

        observed = []

        def respond(request):
            observed.append(json.loads(request.content))
            return httpx.Response(200, json={
                "id": "fixture-response", "object": "chat.completion", "created": 0,
                "model": "moonshotai/kimi-k2-instruct",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "A fact. [1]"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})

        with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
            llm = Groq(model="moonshotai/kimi-k2-instruct", api_key="fixture-key",
                       http_client=http_client, max_retries=0, temperature=0)
            result = answer("What?", [dict(source="manual.txt", text="A fact.")], llm=llm)
        self.assertEqual(result, "A fact. [1]")
        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0]["model"], "moonshotai/kimi-k2-instruct")
        self.assertIn("A fact.", observed[0]["messages"][-1]["content"])

    def test_pipeline_with_real_local_sentence_transformer_and_mock_database(self):
        from sentence_transformers import SentenceTransformer, models

        # A native eight-dimensional bag-of-words model exercises SDK encoding
        # without downloading weights. It does not establish pretrained BGE quality.
        client = self.client()
        client.search.return_value = [[dict(id=0, distance=0,
                                            entity=dict(context="Binary vectors save memory.", source="manual.txt"))]]
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory, "local-model")
            SentenceTransformer(modules=[models.BoW(["binary", "vectors", "save", "memory",
                                                    "query", "one", "bit", "packing"])]).save(str(model_dir))
            Path(directory, "manual.txt").write_text("Binary vectors save memory.", encoding="utf-8")
            with patch("pymilvus.MilvusClient", return_value=client), \
                 patch("socket.socket.connect", side_effect=AssertionError("Unexpected network socket")):
                pipeline = NativePipeline(directory, embed_model=str(model_dir), uri="http://fixture.invalid:19530")
                self.assertEqual(pipeline.retrieve("binary")[0]["source"], "manual.txt")
                pipeline.close()
        self.assertEqual(pipeline.index.dimension, 8)
        self.assertEqual((pipeline.payload_bytes, pipeline.float32_bytes), (1, 32))
        self.assertEqual(len(client.insert.call_args.kwargs["data"][0]["binary_vector"]), 1)
        client.close.assert_called_once()


class StreamlitTests(unittest.TestCase):
    def test_initial_ui_has_no_network_dependency(self):
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            self.skipTest("Streamlit not installed")
        app = AppTest.from_file(str(Path(__file__).with_name("app.py"))).run(timeout=20)
        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, "Binary Quantized RAG")
        self.assertFalse(app.text_input)

    def test_query_flow_with_injected_verified_retrieval(self):
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            self.skipTest("Streamlit not installed")
        pipeline = MagicMock()
        pipeline.index.collection = "test_collection"
        pipeline.payload_bytes, pipeline.float32_bytes = 1, 32
        pipeline.retrieve.return_value = [dict(source="manual.txt", text="A fact.", distance=0)]
        app = AppTest.from_file(str(Path(__file__).with_name("app.py")))
        app.session_state["pipeline"] = pipeline
        with patch("rag.answer", return_value="A fact. [1]") as generate:
            app.run(timeout=20)
            app.text_input[0].set_value("What?").run()
            app.button[0].click().run()
        self.assertFalse(app.exception)
        pipeline.retrieve.assert_called_once_with("What?")
        generate.assert_called_once()
        self.assertIn("A fact. [1]", [m.value for m in app.text])


if __name__ == "__main__":
    unittest.main()
