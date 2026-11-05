import unittest
from unittest.mock import patch
from marketing_probe.api import predict, APIError

class APITest(unittest.TestCase):
    @patch("marketing_probe.api.request")
    def test_generating_no_content_is_actionable_error(self, mock_request):
        mock_request.return_value = ({
            "choices": [{"message": {"content": None}, "finish_reason": "length"}],
            "usage": {"completion_tokens": 120}
        }, 0.2)
        with self.assertRaisesRegex(APIError, "returned no text"):
            predict("luna_chat", "Headline A sample", "Headline B sample")
        self.assertEqual(mock_request.call_args.args[1]["max_tokens"], 1024)

    @patch("marketing_probe.api.request")
    def test_generating_valid_json(self, mock_request):
        mock_request.return_value = ({
            "choices": [{"message": {"content": '{"p_A": 0.3}'}, "finish_reason": "stop"}],
            "usage": {"cost": 0.0001}
        }, 0.2)
        result = predict("luna_chat", "Headline A sample", "Headline B sample")
        self.assertEqual(result["winner"], "B")
        self.assertEqual(result["p_A"], 0.3)
