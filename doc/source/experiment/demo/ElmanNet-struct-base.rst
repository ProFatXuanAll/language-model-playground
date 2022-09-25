Elman Net: structure-related hyperparameters baseline
=====================================================

Abstract
--------
The goal of this experiment is to show how Elman Net language model's structure hyperparameters affect training loss and perplexity.
We found that

- Increasing ``d_emb`` from ``10`` to ``100`` makes training loss and perplexity lower.
- Increasing ``d_hid`` from ``10`` to ``100`` makes training loss and perplexity lower.
- When ``d_emb = 100`` and ``d_hid = 100``, increasing ``n_lyr`` from ``1`` to ``2`` (or ``3``) makes both training loss and perplexity lower.
- Overfitting was observed.
- :math:`100\%` accuracy on training set is almost achieved.
- Performance are really bad for validation set.

Environment setup
-----------------
We ran experiments on Nvidia RTX 2070S.
CUDA version is ``11.4`` and CUDA driver version is ``470.129.06``.

Experiment setup
----------------
We changed the values of ``d_emb``, ``d_hid`` and ``n_lyr`` and recorded training loss and perplexity.
Hyperparameters and their values are listed below.

+-----------+-----------------------+
| Name      | Values                |
+===========+=======================+
| ``d_emb`` | :math:`\set{10, 100}` |
+-----------+-----------------------+
| ``d_hid`` | :math:`\set{10, 100}` |
+-----------+-----------------------+
| ``n_lyr`` | :math:`\set{1, 2, 3}` |
+-----------+-----------------------+

Tokenizer settings
~~~~~~~~~~~~~~~~~~
We used :doc:`lmp.script.train_tknzr </script/train_tknzr>` to train a character tokenizer :py:class:`~lmp.tknzr.CharTknzr`.
Script was executed as below:

.. code-block:: shell

  python -m lmp.script.train_tknzr character \
    --dset_name demo \
    --exp_name demo_tknzr \
    --is_uncased \
    --max_vocab -1 \
    --min_count 0 \
    --ver train

Model training settings
~~~~~~~~~~~~~~~~~~~~~~~
We trained Elman Net language model :py:class:`~lmp.model.ElmanNet` with different model structure hyperparameters.
We used :doc:`lmp.script.train_model </script/train_model>` to train language models.
Script was executed as below:

.. code-block:: shell

  python -m lmp.script.train_model Elman-Net \
    --dset_name demo \
    --batch_size 32 \
    --beta1 0.9 \
    --beta1 0.999 \
    --ckpt_step 500 \
    --d_emb D_EMB \
    --d_hid D_HID \
    --dset_name demo \
    --eps 1e-8 \
    --exp_name EXP_NAME \
    --init_lower -0.1 \
    --init_upper 0.1 \
    --label_smoothing 0.0 \
    --log_step 100 \
    --lr 1e-3 \
    --max_norm 1 \
    --max_seq_len 35 \
    --n_lyr N_LYR \
    --p_emb 0.0 \
    --p_hid 0.0 \
    --seed 42 \
    --stride 35 \
    --tknzr_exp_name demo_tknzr \
    --total_step 30000 \
    --ver train \
    --warmup_step 5000 \
    --weight_decay 0.0

Model evaluation settings
~~~~~~~~~~~~~~~~~~~~~~~~~
We evaluated language models using :doc:`lmp.script.eval_dset_ppl </script/eval_dset_ppl>`.
Script was executed as below:

.. code-block:: shell

  python -m lmp.script.eval_dset_ppl demo \
    --batch_size 512 \
    --exp_name EXP_NAME \
    --first_ckpt 0 \
    --last_ckpt -1 \
    --seed 42 \
    --ver VER

Experiment results
------------------
All results were logged on tensorboard.
You can launch tensorboard with the script

.. code-block:: shell

  pipenv run tensorboard

Training loss
~~~~~~~~~~~~~

+-----------+-----------+-----------+------------+------------+------------+------------+------------+------------+
| ``d_emb`` | ``d_hid`` | ``n_lyr`` | 5k steps   | 10k steps  | 15k steps  | 20k steps  | 25k steps  | 30k steps  |
+===========+===========+===========+============+============+============+============+============+============+
| 10        | 10        | 1         | 0.7045     | 0.4407     | 0.4184     | 0.4081     | 0.4027     | 0.4005     |
+-----------+-----------+-----------+------------+------------+------------+------------+------------+------------+
| 10        | 10        | 2         | 1.347      | 0.4885     | 0.434      | 0.4289     | 0.4249     | 0.4241     |
+-----------+-----------+-----------+------------+------------+------------+------------+------------+------------+
| 10        | 10        | 3         | 2.502      | 0.5185     | 0.4507     | 0.4363     | 0.4298     | 0.4261     |
+-----------+-----------+-----------+------------+------------+------------+------------+------------+------------+
| 10        | 100       | 1         | 0.516      | 0.3896     | 0.3654     | 0.3526     | 0.3442     | 0.3417     |
+-----------+-----------+-----------+------------+------------+------------+------------+------------+------------+
| 10        | 100       | 2         | 0.8442     | 0.4833     | 0.4291     | 0.41       | 0.3787     | 0.3706     |
+-----------+-----------+-----------+------------+------------+------------+------------+------------+------------+
| 10        | 100       | 3         | 0.4889     | 0.4062     | 0.3715     | 0.3536     | 0.3411     | 0.3327     |
+-----------+-----------+-----------+------------+------------+------------+------------+------------+------------+
| 100       | 10        | 1         | 0.4237     | 0.4073     | 0.3728     | 0.3618     | 0.3562     | 0.354      |
+-----------+-----------+-----------+------------+------------+------------+------------+------------+------------+
| 100       | 10        | 2         | 0.4274     | 0.4161     | 0.3879     | 0.3754     | 0.3674     | 0.3646     |
+-----------+-----------+-----------+------------+------------+------------+------------+------------+------------+
| 100       | 10        | 3         | 0.4249     | 0.4152     | 0.4131     | 0.4123     | 0.4114     | 0.3976     |
+-----------+-----------+-----------+------------+------------+------------+------------+------------+------------+
| 100       | 100       | 1         | 0.3422     | 0.3122     | 0.3016     | 0.2907     | 0.2812     | 0.2775     |
+-----------+-----------+-----------+------------+------------+------------+------------+------------+------------+
| 100       | 100       | 2         | 0.333      | **0.3025** | **0.2928** | **0.2821** | 0.2712     | 0.2651     |
+-----------+-----------+-----------+------------+------------+------------+------------+------------+------------+
| 100       | 100       | 3         | **0.3313** | 0.3068     | 0.2939     | 0.2846     | **0.2678** | **0.2611** |
+-----------+-----------+-----------+------------+------------+------------+------------+------------+------------+

Observation 1: Increasing ``d_emb`` from ``10`` to ``100`` makes training loss smaller.
***************************************************************************************
By fixing ``d_hid`` and ``n_lyr``, we can compare training loss for ``d_emb = 10`` and ``d_emb = 100``.
All comparisons (:math:`\dfrac{36}{36}`) show that training loss is smaller when increasing ``d_emb`` from ``10`` to ``100``.

Observation 2: Increasing ``d_hid`` from ``10`` to ``100`` makes training loss smaller.
***************************************************************************************
By fixing ``d_emb`` and ``n_lyr``, we can compare training loss for ``d_hid = 10`` and ``d_hid = 100``.
All comparisons (:math:`\dfrac{36}{36})` show that training loss is smaller when increasing ``d_hid`` from ``10`` to ``100``.

Observation 3: When ``d_emb = 10``, increasing ``n_lyr`` from ``1`` to ``2`` makes training loss larger.
********************************************************************************************************
By fixing ``d_emb = 10`` and ``d_hid``, we can compare training loss for ``n_lyr = 1`` and ``n_lyr = 2``.
All comparisons (:math:`\dfrac{12}{12})` show that training loss is larger when increasing ``n_lyr`` from ``1`` to ``2``.

Observation 4: When ``d_emb = 10``, increasing ``n_lyr`` from ``1`` to ``3`` in general makes training loss larger.
*******************************************************************************************************************
By fixing ``d_emb = 10`` and ``d_hid``, we can compare training loss for ``n_lyr = 1`` and ``n_lyr = 3``.
:math:`9` out of :math:`12` comparisons show that training loss is larger when increasing ``n_lyr`` from ``1`` to ``3``.

Observation 5: When ``d_emb = 100`` and ``d_hid = 10``, increasing ``n_lyr`` from ``1`` to ``2`` makes training loss larger.
****************************************************************************************************************************
By fixing ``d_emb = 100`` and ``d_hid = 10``, we can compare training loss for ``n_lyr = 1`` and ``n_lyr = 2``.
All comparisons (:math:`\dfrac{6}{6})` show that training loss is larger when increasing ``n_lyr`` from ``1`` to ``2``.

Observation 6: When ``d_emb = 100`` and ``d_hid = 100``, increasing ``n_lyr`` from ``1`` to ``2`` makes training loss smaller.
******************************************************************************************************************************
By fixing ``d_emb = 100`` and ``d_hid = 100``, we can compare training loss for ``n_lyr = 1`` and ``n_lyr = 2``.
All comparisons (:math:`\dfrac{6}{6})` show that training loss is smaller when increasing ``n_lyr`` from ``1`` to ``2``.
One should compare this with observation 5.

Observation 7: When ``d_emb = 100`` and ``d_hid = 10``, increasing ``n_lyr`` from ``1`` to ``3`` makes training loss larger.
****************************************************************************************************************************
By fixing ``d_emb = 100`` and ``d_hid = 10``, we can compare training loss for ``n_lyr = 1`` and ``n_lyr = 3``.
All comparisons (:math:`\dfrac{6}{6})` show that training loss is larger when increasing ``n_lyr`` from ``1`` to ``3``.

Observation 8: When ``d_emb = 100`` and ``d_hid = 100``, increasing ``n_lyr`` from ``1`` to ``3`` makes training loss larger.
*****************************************************************************************************************************
By fixing ``d_emb = 100`` and ``d_hid = 100``, we can compare training loss for ``n_lyr = 1`` and ``n_lyr = 3``.
All comparisons (:math:`\dfrac{6}{6})` show that training loss is smaller when increasing ``n_lyr`` from ``1`` to ``3``.
One should compare this with observation 7.

Observation 9: Increasing ``n_lyr`` must also increase ``d_emb`` and ``d_hid``.
*******************************************************************************
Combining observations from 3 to 9, we conclude that when increasing ``n_lyr``, one have to increase ``d_emb`` and ``d_hid`` together to make training loss smaller.

Observation 10: Minimum loss is achieved when ``d_emb = 100``, ``d_hid = 100`` and ``n_lyr = 3``.
*************************************************************************************************

Observation 11: Training loss is still decreasing in all configuration.
***********************************************************************
All comparisons (:math:`\dfrac{60}{60}`) show that training loss is still decreasing no matter which configuration is used.
This suggest that further training may be required.

Perplexity
~~~~~~~~~~

+-----------+-----------+-----------+----------------------------------+----------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+
| ``d_emb`` | ``d_hid`` | ``n_lyr`` | 5k steps                         | 10k steps                        | 15k steps                         | 20k steps                         | 25k steps                         | 30k steps                         |
|           |           |           +----------+-----------+-----------+-----------+-----------+----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
|           |           |           | train    | valid     | test      | train     | valid     | test     | train     | valid     | test      | train     | valid     | test      | train     | valid     | test      | train     | valid     | test      |
+===========+===========+===========+==========+===========+===========+===========+===========+==========+===========+===========+===========+===========+===========+===========+===========+===========+===========+===========+===========+===========+
| 10        | 10        | 1         | 1.976    | 2.017     | 2.009     | 1.533     | 1.649     | 1.591    | 1.502     | 1.606     | 1.566     | 1.486     | 1.608     | 1.551     | 1.478     | 1.604     | 1.545     | 1.476     | 1.605     | 1.543     |
+-----------+-----------+-----------+----------+-----------+-----------+-----------+-----------+----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| 10        | 10        | 2         | 3.566    | 3.669     | 3.642     | 1.604     | 1.634     | 1.63     | 1.524     | **1.55**  | 1.549     | 1.516     | **1.559** | 1.55      | 1.511     | **1.571** | 1.551     | 1.51      | **1.588** | 1.553     |
+-----------+-----------+-----------+----------+-----------+-----------+-----------+-----------+----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| 10        | 10        | 3         | 11.34    | 11.43     | 11.35     | 1.653     | 1.693     | 1.686    | 1.547     | 1.586     | 1.585     | 1.527     | 1.574     | 1.572     | 1.518     | 1.594     | 1.575     | 1.513     | 1.594     | 1.571     |
+-----------+-----------+-----------+----------+-----------+-----------+-----------+-----------+----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| 10        | 100       | 1         | 1.638    | 2.223     | 1.699     | 1.455     | 1.774     | 1.515    | 1.423     | 1.861     | 1.485     | 1.41      | 1.992     | 1.466     | 1.398     | 2.145     | 1.457     | 1.393     | 2.148     | 1.451     |
+-----------+-----------+-----------+----------+-----------+-----------+-----------+-----------+----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| 10        | 100       | 2         | 2.243    | 3.267     | 2.284     | 1.597     | **1.633** | 1.636    | 1.516     | 1.631     | 1.555     | 1.487     | 1.667     | 1.526     | 1.449     | 1.697     | 1.498     | 1.433     | 1.717     | 1.49      |
+-----------+-----------+-----------+----------+-----------+-----------+-----------+-----------+----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| 10        | 100       | 3         | 1.602    | 2.306     | 1.622     | 1.474     | 1.676     | 1.514    | 1.429     | 1.785     | 1.478     | 1.408     | 1.87      | 1.475     | 1.392     | 1.932     | 1.46      | 1.381     | 1.912     | 1.441     |
+-----------+-----------+-----------+----------+-----------+-----------+-----------+-----------+----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| 100       | 10        | 1         | 1.507    | **1.717** | 1.566     | 1.483     | 1.759     | 1.533    | 1.436     | 1.852     | 1.493     | 1.423     | 1.898     | 1.477     | 1.415     | 1.921     | 1.472     | 1.41      | 1.948     | 1.471     |
+-----------+-----------+-----------+----------+-----------+-----------+-----------+-----------+----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| 100       | 10        | 2         | 1.515    | 1.74      | 1.568     | 1.498     | 1.681     | 1.553    | 1.457     | 1.804     | 1.524     | 1.439     | 1.799     | 1.512     | 1.43      | 1.804     | 1.502     | 1.424     | 1.797     | 1.495     |
+-----------+-----------+-----------+----------+-----------+-----------+-----------+-----------+----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| 100       | 10        | 3         | 1.51     | 1.79      | 1.586     | 1.496     | 1.709     | 1.562    | 1.493     | 1.795     | 1.576     | 1.492     | 1.875     | 1.574     | 1.491     | 1.926     | 1.565     | 1.47      | 1.945     | 1.53      |
+-----------+-----------+-----------+----------+-----------+-----------+-----------+-----------+----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| 100       | 100       | 1         | 1.401    | 1.939     | 1.458     | 1.349     | 2.489     | 1.422    | 1.344     | 3.035     | 1.417     | 1.323     | 3.435     | 1.391     | 1.315     | 3.733     | 1.39      | 1.309     | 3.867     | 1.392     |
+-----------+-----------+-----------+----------+-----------+-----------+-----------+-----------+----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| 100       | 100       | 2         | **1.377**| 2.103     | **1.438** | 1.345     | 3.38      | **1.405**| 1.326     | 4.785     | 1.411     | 1.316     | 5.542     | 1.407     | 1.302     | 6.486     | 1.398     | 1.294     | 6.949     | 1.377     |
+-----------+-----------+-----------+----------+-----------+-----------+-----------+-----------+----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| 100       | 100       | 3         | **1.377**| 1.932     | 1.486     | **1.342** | 2.692     | 1.406    | **1.324** | 3.359     | **1.376** | **1.314** | 4.503     | **1.388** | **1.299** | 4.526     | **1.36**  | **1.288** | 4.691     | **1.372** |
+-----------+-----------+-----------+----------+-----------+-----------+-----------+-----------+----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+

Observation 1: Increasing ``d_emb`` from ``10`` to ``100`` makes perplexity smaller.
************************************************************************************
By fixing ``d_hid`` and ``n_lyr``, we can compare perplexity for ``d_emb = 10`` and ``d_emb = 100``.
Most of the comparisons (:math:`\dfrac{77}{108}`) show that perplexity is smaller when increasing ``d_emb`` from ``10`` to ``100``.

Observation 2: Increasing ``d_hid`` from ``10`` to ``100`` makes perplexity smaller.
************************************************************************************
By fixing ``d_emb`` and ``n_lyr``, we can compare perplexity for ``d_hid = 10`` and ``d_hid = 100``.
Most of the comparisons (:math:`\dfrac{75}{108}`) show that perplexity is smaller when increasing ``d_hid`` from ``10`` to ``100``.

Observation 3: When ``d_emb = 10``, increasing ``n_lyr`` from ``1`` to ``2`` in general makes perplexity larger.
****************************************************************************************************************
By fixing ``d_emb = 10`` and ``d_hid``, we can compare perplexity for ``n_lyr = 1`` and ``n_lyr = 2``.
Most of the comparisons (:math:`\dfrac{24}{36}`) show that perplexity is larger when increasing ``n_lyr`` from ``1`` to ``2``.

Observation 4: When ``d_emb = 10``, increasing ``n_lyr`` from ``1`` to ``3`` does not show anything significant.
****************************************************************************************************************
By fixing ``d_emb = 10`` and ``d_hid``, we can compare perplexity for ``n_lyr = 1`` and ``n_lyr = 3``.
About half of comparisons (:math:`\dfrac{19}{36}`) show that perplexity is larger when increasing ``n_lyr`` from ``1`` to ``3``.
No significance was shown and no conclusion could be made.

Observation 5: When ``d_emb = 100``, increasing ``n_lyr`` from ``1`` to ``2`` in general makes perplexity larger.
*****************************************************************************************************************
By fixing ``d_emb = 100`` and ``d_hid``, we can compare perplexity for ``n_lyr = 1`` and ``n_lyr = 2``.
Most of the comparisons (:math:`\dfrac{21}{36}`) show that perplexity is smaller when increasing ``n_lyr`` from ``1`` to ``2``.

Observation 6: When ``d_emb = 100``, increasing ``n_lyr`` from ``1`` to ``3`` does not show anything significant.
*****************************************************************************************************************
By fixing ``d_emb = 100`` and ``d_hid``, we can compare perplexity for ``n_lyr = 1`` and ``n_lyr = 3``.
About half of comparisons (:math:`\dfrac{20}{36}`) show that perplexity is smaller when increasing ``n_lyr`` from ``1`` to ``3``.
No significance was shown and no conclusion could be made.

Observation 7: Overfitting seems to happen.
*******************************************
On test set, most comparisons (:math:`\dfrac{53}{60}`) show that perplexity is still decreasing.
However, on validation set, most comparisons (:math:`\dfrac{42}{60}`) show that perplexity is increasing.
Perplexity on validation set increase early, most of them happened at either ``10k`` or ``15k`` steps.

Observation 8: Minimum perplexity on training set is achieved at ``30k`` step when ``d_emb = 100``, ``d_hid = 100`` and ``n_lyr = 3``.
**************************************************************************************************************************************
- On training set, minimum perplexity :math:`1.288` is achieved at ``30k`` step when ``d_emb = 100``, ``d_hid = 100`` and ``n_lyr = 3``.
- On validation set, minimum perplexity :math:`1.55` is achieved at ``15k`` step when ``d_emb = 10``, ``d_hid = 10`` and ``n_lyr = 2``.
- On testing set, minimum perplexity :math:`1.36` is achieved at ``25k`` step when ``d_emb = 100``, ``d_hid = 100`` and ``n_lyr = 3``.

Observation 9: Only when setting ``d_emb = 100`` and ``d_hid = 100`` perplexity is less than :math:`1.4`.
*********************************************************************************************************
Later in the accuracy experiments we see that training set accuracy is higher than :math:`90\%` only when perplexity is less than :math:`1.4`.

Accuracy
~~~~~~~~
We use the following script to calculate accuracy on demo dataset:

.. code-block:: python

  import re

  import torch

  import lmp.dset
  import lmp.infer
  import lmp.model
  import lmp.script
  import lmp.tknzr
  import lmp.util.model
  import lmp.util.tknzr

  device = torch.device('cuda')
  tknzr = lmp.util.tknzr.load(exp_name='demo_tknzr')
  for d_emb in [10, 100]:
    for d_hid in [10, 100]:
      for n_lyr in [1, 2, 3]:
        for ckpt in [5000, 10000, 15000, 20000, 25000, 30000]:
          for ver in lmp.dset.DemoDset.vers:
            dset = lmp.dset.DemoDset(ver=ver)
            exp_name = f'demo-d_emb-{d_emb}-d_hid-{d_hid}-n_lyr-{n_lyr}'
            model = lmp.util.model.load(exp_name=exp_name, ckpt=ckpt).to(device)
            infer = lmp.infer.Top1Infer(max_seq_len=35)

            correct = 0
            for spl in dset:
              match = re.match(r'If you add (\d+) to (\d+) you get (\d+) .', spl)
              input = f'If you add {match.group(1)} to {match.group(2)} you get '

              output = infer.gen(model=model, tknzr=tknzr, txt=input)

              if input + output == spl:
                correct += 1

            print(f'{exp_name}, ckpt: {ckpt}, ver: {ver}, acc: {correct / len(dset) * 100 :.2f}%')


+-----------+-----------+-----------+-------------------------------+-------------------------------+-------------------------------+-------------------------------+------------------------------+-----------------------------+
| ``d_emb`` | ``d_hid`` | ``n_lyr`` | 5k steps                      | 10k steps                     | 15k steps                     | 20k steps                     | 25k steps                    | 30k steps                   |
|           |           |           +-----------+-----------+-------+-----------+-----------+-------+-----------+-----------+-------+-----------+-----------+-------+-----------+----------+-------+----------+----------+-------+
|           |           |           | train     | valid     | test  | train     | valid     | test  | train     | valid     | test  | train     | valid     | test  | train     | valid    | test  | train    | valid    | test  |
+===========+===========+===========+===========+===========+=======+===========+===========+=======+===========+===========+=======+===========+===========+=======+===========+==========+=======+==========+==========+=======+
| 10        | 10        | 1         | 0.99      | 0.99      | 0     | 1.09      | 0.63      | 1     | 0.99      | 1.03      | 0     | 1.58      | 1.15      | 0     | 2.36      | 1.54     | 1     | 2.3      | 1.62     | 2     |
+-----------+-----------+-----------+-----------+-----------+-------+-----------+-----------+-------+-----------+-----------+-------+-----------+-----------+-------+-----------+----------+-------+----------+----------+-------+
| 10        | 10        | 2         | 0.89      | 0.89      | 0     | 0.89      | 0.89      | 0     | 0.89      | 0.89      | 1     | 0.99      | 0.99      | 1     | 0.99      | 0.99     | 1     | 0.99     | 0.99     | 1     |
+-----------+-----------+-----------+-----------+-----------+-------+-----------+-----------+-------+-----------+-----------+-------+-----------+-----------+-------+-----------+----------+-------+----------+----------+-------+
| 10        | 10        | 3         | 0         | 0         | 0     | 0.99      | 0.99      | 1     | 0.99      | 0.99      | 1     | 0.99      | 0.99      | 1     | 0.99      | 0.99     | 0     | 0.99     | 0.99     | 0     |
+-----------+-----------+-----------+-----------+-----------+-------+-----------+-----------+-------+-----------+-----------+-------+-----------+-----------+-------+-----------+----------+-------+----------+----------+-------+
| 10        | 100       | 1         | 0.99      | 0.99      | 1     | 3.6       | 1.39      | 1     | 9.68      | 2.79      | 6     | 11.13     | 3.45      | 6     | 21.17     | 5.19     | 13    | 21.72    | 5.19     | 11    |
+-----------+-----------+-----------+-----------+-----------+-------+-----------+-----------+-------+-----------+-----------+-------+-----------+-----------+-------+-----------+----------+-------+----------+----------+-------+
| 10        | 100       | 2         | 0         | 0         | 0     | 0.91      | 0.91      | 1     | 0.91      | 0.91      | 0     | 0.99      | 0.53      | 1     | 5.94      | 2.2      | 4     | 9.62     | 3.15     | 4     |
+-----------+-----------+-----------+-----------+-----------+-------+-----------+-----------+-------+-----------+-----------+-------+-----------+-----------+-------+-----------+----------+-------+----------+----------+-------+
| 10        | 100       | 3         | 1.13      | 0.89      | 0     | 3.72      | 2.51      | 1     | 13.07     | 2.61      | 3     | 16.73     | 4.79      | 5     | 28.61     | 7.41     | 13    | 41.29    | 7.62     | 28    |
+-----------+-----------+-----------+-----------+-----------+-------+-----------+-----------+-------+-----------+-----------+-------+-----------+-----------+-------+-----------+----------+-------+----------+----------+-------+
| 100       | 10        | 1         | 0.99      | 0.99      | 0     | 1.07      | 0.61      | 1     | 4.26      | 1.76      | 2     | 5.72      | 1.96      | 4     | 6.75      | 3.21     | 4     | 7.54     | 4.1      | 1     |
+-----------+-----------+-----------+-----------+-----------+-------+-----------+-----------+-------+-----------+-----------+-------+-----------+-----------+-------+-----------+----------+-------+----------+----------+-------+
| 100       | 10        | 2         | 0.1       | 0.1       | 1     | 1.05      | 0.12      | 2     | 2.12      | 0.89      | 4     | 6.14      | 1.9       | 3     | 6.95      | 1.76     | 4     | 10.2     | 2.59     | 10    |
+-----------+-----------+-----------+-----------+-----------+-------+-----------+-----------+-------+-----------+-----------+-------+-----------+-----------+-------+-----------+----------+-------+----------+----------+-------+
| 100       | 10        | 3         | 0.89      | 0.89      | 0     | 0.95      | 0.95      | 1     | 1.01      | 0.95      | 1     | 0.97      | 0.97      | 1     | 0.97      | 0.93     | 1     | 1.58     | 1.07     | 1     |
+-----------+-----------+-----------+-----------+-----------+-------+-----------+-----------+-------+-----------+-----------+-------+-----------+-----------+-------+-----------+----------+-------+----------+----------+-------+
| 100       | 100       | 1         | 8.61      | 2.16      | 5     | 35.31     | 7.05      | 18    | 34.14     | 4.77      | 18    | 65.76     | 7.43      | 40    | 87.09     | 6.99     | 52    | 92.89    | 7.27     | 60    |
+-----------+-----------+-----------+-----------+-----------+-------+-----------+-----------+-------+-----------+-----------+-------+-----------+-----------+-------+-----------+----------+-------+----------+----------+-------+
| 100       | 100       | 2         | 16.97     | 6.97      | **14**| 30.51     | 6.97      | **19**| **58.08** | 6.3       | 29    | 65.54     | 7.68      | 43    | 96.34     | 9.39     | 75    | **99.72**| 11.49    | 83    |
+-----------+-----------+-----------+-----------+-----------+-------+-----------+-----------+-------+-----------+-----------+-------+-----------+-----------+-------+-----------+----------+-------+----------+----------+-------+
| 100       | 100       | 3         | **19.25** | **7.47**  | 6     | **36.12** | **11.8**  | 18    | 51.64     | **9.7**   | **41**| **67.8**  | **9.98**  | **48**| **97.9**  | **13.56**| **78**| 99.6     | **18.02**| **92**|
+-----------+-----------+-----------+-----------+-----------+-------+-----------+-----------+-------+-----------+-----------+-------+-----------+-----------+-------+-----------+----------+-------+----------+----------+-------+

Observation 1: :math:`100\%` accuracy is not achieved on training set.
**********************************************************************
The highest accuracy can be achieved on training set is :math:`99.72\%`.
:math:`99.72\%` accuracy is achieved using ``d_emb = 100``, ``d_hid = 100`` and ``n_lyr = 2``.

Observation 2: :math:`100\%` accuracy is not achieved on test set.
******************************************************************
The highest accuracy can be achieved on test set is :math:`92\%`.
:math:`92\%` accuracy is achieved using ``d_emb = 100``, ``d_hid = 100`` and ``n_lyr = 3``.
One should compare this with observation 1.

Observation 3: Accuracy on validation set is less than :math:`20\%`.
********************************************************************
The highest accuracy can be achieved on validation set is :math:`18.02\%`.
This happened when the best accuracy is achieved on test set (see observation 2).

Observation 4: Commutative law for addition seems to be harder to generalized than reflexive addition.
******************************************************************************************************
Validation set is basically training set but changing ``a + b`` to ``b + a``.
Test set is only consist of ``a + a``.
From observation 2 and 3 we know that model generalized well on test set but not validation set.

Future work
-----------
Find a way to make model generalize on validation set.

.. footbibliography::
