from __future__ import annotations

import unittest

from voice_assistant.http_utils import validated_http_url


class HttpUrlValidationTests(unittest.TestCase):
    def test_accepts_local_http_and_strips_trailing_slash(self) -> None:
        self.assertEqual(
            validated_http_url("http://127.0.0.1:11434/"),
            "http://127.0.0.1:11434",
        )

    def test_rejects_file_urls_and_embedded_credentials(self) -> None:
        with self.assertRaises(ValueError):
            validated_http_url("file:///tmp/private")
        with self.assertRaises(ValueError):
            validated_http_url("https://user:password@example.com")

    def test_https_only_rejects_plain_http(self) -> None:
        with self.assertRaises(ValueError):
            validated_http_url("http://api.example.com", require_https=True)


if __name__ == "__main__":
    unittest.main()
