r"""Test `lmp.util.generate_sequence.`.

Usage:
    python -m unittest test.lmp.util._generate_sequence.test_generate_sequence
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import gc
import inspect
import math
import unittest

from typing import List
from typing import Union

# 3rd-party modules

import torch

# self-made modules

import lmp.model
import lmp.util


class TestGenerateSequence(unittest.TestCase):
    r"""Test case for `lmp.util.generate_sequence`."""

    def setUp(self):
        r"""Setup fixed parameters."""
        self.beam_width = 1
        self.begin_of_sequence = ''
        self.device = torch.device('cpu')
        self.max_seq_len = 2
        self.model = lmp.model.BaseRNNModel(
            d_emb=1,
            d_hid=1,
            dropout=0.0,
            num_linear_layers=1,
            num_rnn_layers=1,
            pad_token_id=0,
            vocab_size=5
        )
        self.tokenizer = lmp.tokenizer.CharDictTokenizer()

    def tearDown(self):
        r"""Delete fixed parameters."""
        del self.beam_width
        del self.begin_of_sequence
        del self.device
        del self.max_seq_len
        del self.model
        del self.tokenizer
        gc.collect()

    def test_signature(self):
        r"""Ensure signature consistency."""
        msg = 'Inconsistent method signature.'

        self.assertEqual(
            inspect.signature(lmp.util.generate_sequence),
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name='beam_width',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=int,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='begin_of_sequence',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=str,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='device',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=torch.device,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='max_seq_len',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=int,
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='model',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=Union[
                            lmp.model.BaseRNNModel,
                            lmp.model.BaseResRNNModel,
                        ],
                        default=inspect.Parameter.empty
                    ),
                    inspect.Parameter(
                        name='tokenizer',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=lmp.tokenizer.BaseTokenizer,
                        default=inspect.Parameter.empty
                    )
                ],
                return_annotation=List[str]
            ),
            msg=msg
        )

    def test_invalid_input_beam_width(self):
        r"""Raise exception when input `beam_width` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when input `beam_width` '
            'is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            False, 0, 0.0, 1.0, math.nan, -math.nan, math.inf, -math.inf, 0j,
            1j, '', b'', (), [], {}, set(), object(), lambda x: x, type, None,
            NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(
                    (TypeError, ValueError),
                    msg=msg1
            ) as ctx_man:
                lmp.util.generate_sequence(
                    beam_width=invalid_input,
                    begin_of_sequence=self.begin_of_sequence,
                    device=self.device,
                    max_seq_len=self.max_seq_len,
                    model=self.model,
                    tokenizer=self.tokenizer
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`beam_width` must be an instance of `int`.',
                    msg=msg2
                )
            else:
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`beam_width` must be bigger than or equal to `1`.',
                    msg=msg2
                )

    def test_invalid_input_begin_of_sequence(self):
        r"""Raise `TypeError` when input `begin_of_sequence` is invalid."""
        msg1 = (
            'Must raise `TypeError` when input `begin_of_sequence` is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, b'', (), [], {}, set(), object(), lambda x: x,
            type, None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.generate_sequence(
                    beam_width=self.beam_width,
                    begin_of_sequence=invalid_input,
                    device=self.device,
                    max_seq_len=self.max_seq_len,
                    model=self.model,
                    tokenizer=self.tokenizer
                )

            self.assertEqual(
                ctx_man.exception.args[0],
                '`begin_of_sequence` must be an instance of `str`.',
                msg=msg2
            )

    def test_invalid_input_device(self):
        r"""Raise `TypeError` when input `device` is invalid."""
        msg1 = 'Must raise `TypeError` when input `device` is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, '', b'', (), [], {}, set(), object(),
            lambda x: x, type, None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.generate_sequence(
                    beam_width=self.beam_width,
                    begin_of_sequence=self.begin_of_sequence,
                    device=invalid_input,
                    max_seq_len=self.max_seq_len,
                    model=self.model,
                    tokenizer=self.tokenizer
                )

            self.assertEqual(
                ctx_man.exception.args[0],
                '`device` must be an instance of `torch.device`.',
                msg=msg2
            )

    def test_invalid_input_max_seq_len(self):
        r"""Raise exception when input `max_seq_len` is invalid."""
        msg1 = (
            'Must raise `TypeError` or `ValueError` when input `max_seq_len` '
            'is invalid.'
        )
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, '', b'', (), [], {}, set(), object(),
            lambda x: x, type, None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(
                    (TypeError, ValueError),
                    msg=msg1
            ) as ctx_man:
                lmp.util.generate_sequence(
                    beam_width=self.beam_width,
                    begin_of_sequence=self.begin_of_sequence,
                    device=self.device,
                    max_seq_len=invalid_input,
                    model=self.model,
                    tokenizer=self.tokenizer
                )

            if isinstance(ctx_man.exception, TypeError):
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`max_seq_len` must be an instance of `int`.',
                    msg=msg2
                )
            else:
                self.assertEqual(
                    ctx_man.exception.args[0],
                    '`max_seq_len` must be bigger than or equal to `2`.',
                    msg=msg2
                )

    def test_invalid_input_model(self):
        r"""Raise `TypeError` when input `model` is invalid."""
        msg1 = 'Must raise `TypeError` when input `model` is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, '', b'', (), [], {}, set(), object(),
            lambda x: x, type, None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.generate_sequence(
                    beam_width=self.beam_width,
                    begin_of_sequence=self.begin_of_sequence,
                    device=self.device,
                    max_seq_len=self.max_seq_len,
                    model=invalid_input,
                    tokenizer=self.tokenizer
                )

            self.assertEqual(
                ctx_man.exception.args[0],
                '`model` must be an instance of '
                '`Union[lmp.model.BaseRNNModel, lmp.model.BaseResRNNModel]`.',
                msg=msg2
            )

    def test_invalid_input_tokenizer(self):
        r"""Raise `TypeError` when input `tokenizer` is invalid."""
        msg1 = 'Must raise `TypeError` when input `tokenizer` is invalid.'
        msg2 = 'Inconsistent error message.'
        examples = (
            False, True, 0, 1, -1, 0.0, 1.0, math.nan, -math.nan, math.inf,
            -math.inf, 0j, 1j, '', b'', (), [], {}, set(), object(),
            lambda x: x, type, None, NotImplemented, ...
        )

        for invalid_input in examples:
            with self.assertRaises(TypeError, msg=msg1) as ctx_man:
                lmp.util.generate_sequence(
                    beam_width=self.beam_width,
                    begin_of_sequence=self.begin_of_sequence,
                    device=self.device,
                    max_seq_len=self.max_seq_len,
                    model=self.model,
                    tokenizer=invalid_input
                )

            self.assertEqual(
                ctx_man.exception.args[0],
                '`tokenizer` must be an instance of '
                '`lmp.tokenizer.BaseTokenizer`.',
                msg=msg2
            )

    def test_return_type(self):
        r"""Return `List[str]`."""
        msg = 'Must return `List[str]`.'
        examples = (
            (
                self.beam_width,
                self.begin_of_sequence,
                self.device,
                self.max_seq_len,
                self.model,
                self.tokenizer,
            ),
            (
                2,
                '',
                torch.device('cpu'),
                3,
                lmp.model.GRUModel(
                    d_emb=1,
                    d_hid=1,
                    dropout=0.0,
                    num_linear_layers=1,
                    num_rnn_layers=1,
                    pad_token_id=0,
                    vocab_size=10
                ),
                lmp.tokenizer.CharListTokenizer(),
            ),
            (
                3,
                '',
                torch.device('cpu'),
                4,
                lmp.model.LSTMModel(
                    d_emb=1,
                    d_hid=1,
                    dropout=0.0,
                    num_linear_layers=1,
                    num_rnn_layers=1,
                    pad_token_id=0,
                    vocab_size=10
                ),
                lmp.tokenizer.WhitespaceDictTokenizer(),
            ),
            (
                4,
                '',
                torch.device('cpu'),
                5,
                lmp.model.BaseResRNNModel(
                    d_emb=1,
                    d_hid=1,
                    dropout=0.0,
                    num_linear_layers=1,
                    num_rnn_layers=1,
                    pad_token_id=0,
                    vocab_size=10
                ),
                lmp.tokenizer.WhitespaceListTokenizer(),
            ),
            (
                5,
                '',
                torch.device('cpu'),
                6,
                lmp.model.ResGRUModel(
                    d_emb=1,
                    d_hid=1,
                    dropout=0.0,
                    num_linear_layers=1,
                    num_rnn_layers=1,
                    pad_token_id=0,
                    vocab_size=10
                ),
                lmp.tokenizer.CharDictTokenizer(),
            ),
            (
                6,
                '',
                torch.device('cpu'),
                7,
                lmp.model.ResLSTMModel(
                    d_emb=1,
                    d_hid=1,
                    dropout=0.0,
                    num_linear_layers=1,
                    num_rnn_layers=1,
                    pad_token_id=0,
                    vocab_size=10
                ),
                lmp.tokenizer.CharDictTokenizer(is_uncased=True),
            ),
        )

        for (
                beam_width,
                begin_of_sequence,
                device,
                max_seq_len,
                model,
                tokenizer
        ) in examples:
            generated_sequences = lmp.util.generate_sequence(
                beam_width=beam_width,
                begin_of_sequence=begin_of_sequence,
                device=device,
                max_seq_len=max_seq_len,
                model=model,
                tokenizer=tokenizer
            )
            self.assertIsInstance(generated_sequences, list, msg=msg)
            for sequence in generated_sequences:
                self.assertIsInstance(sequence, str, msg=msg)

    def test_return_result(self):
        r"""Return `beam_width` sequences with length `max_seq_len`."""
        msg = 'Must return `beam_width` sequences with length `max_seq_len`.'
        examples = (
            (
                self.beam_width,
                self.begin_of_sequence,
                self.device,
                self.max_seq_len,
                self.model,
                self.tokenizer,
            ),
            (
                2,
                '',
                torch.device('cpu'),
                3,
                lmp.model.GRUModel(
                    d_emb=1,
                    d_hid=1,
                    dropout=0.0,
                    num_linear_layers=1,
                    num_rnn_layers=1,
                    pad_token_id=0,
                    vocab_size=10
                ),
                lmp.tokenizer.CharListTokenizer(),
            ),
            (
                3,
                '',
                torch.device('cpu'),
                4,
                lmp.model.LSTMModel(
                    d_emb=1,
                    d_hid=1,
                    dropout=0.0,
                    num_linear_layers=1,
                    num_rnn_layers=1,
                    pad_token_id=0,
                    vocab_size=10
                ),
                lmp.tokenizer.WhitespaceDictTokenizer(),
            ),
            (
                4,
                '',
                torch.device('cpu'),
                5,
                lmp.model.BaseResRNNModel(
                    d_emb=1,
                    d_hid=1,
                    dropout=0.0,
                    num_linear_layers=1,
                    num_rnn_layers=1,
                    pad_token_id=0,
                    vocab_size=10
                ),
                lmp.tokenizer.WhitespaceListTokenizer(),
            ),
            (
                5,
                '',
                torch.device('cpu'),
                6,
                lmp.model.ResGRUModel(
                    d_emb=1,
                    d_hid=1,
                    dropout=0.0,
                    num_linear_layers=1,
                    num_rnn_layers=1,
                    pad_token_id=0,
                    vocab_size=10
                ),
                lmp.tokenizer.CharDictTokenizer(),
            ),
            (
                6,
                '',
                torch.device('cpu'),
                7,
                lmp.model.ResLSTMModel(
                    d_emb=1,
                    d_hid=1,
                    dropout=0.0,
                    num_linear_layers=1,
                    num_rnn_layers=1,
                    pad_token_id=0,
                    vocab_size=10
                ),
                lmp.tokenizer.CharDictTokenizer(is_uncased=True),
            ),
        )

        for (
                beam_width,
                begin_of_sequence,
                device,
                max_seq_len,
                model,
                tokenizer
        ) in examples:
            generated_sequences = lmp.util.generate_sequence(
                beam_width=beam_width,
                begin_of_sequence=begin_of_sequence,
                device=device,
                max_seq_len=max_seq_len,
                model=model,
                tokenizer=tokenizer
            )
            self.assertEqual(len(generated_sequences), beam_width, msg=msg)
            for sequence in generated_sequences:
                self.assertEqual(
                    len(sequence),
                    len(tokenizer.detokenize(['[unk]'] * max_seq_len)),
                    msg=msg
                )


if __name__ == '__main__':
    unittest.main()
