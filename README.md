# IMDB-Review-Sentiment-NLP-PyTorch-BERT
Using pytorch and natural language processing techniques to classify positive and negative IMDB reviews as well as custom reviews. Achieved 88% validation accuracy on custom LSTM and 94% on BERT!

## Debugging

### 1. KeyError 4605 (or any other number)

**Example:** 
```python
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
File c:\Users\PanSt\Desktop\4ML\Sarcasm-Detection-using-NLP\.venv\Lib\site-packages\pandas\core\indexes\base.py:3805, in Index.get_loc(self, key)
   3804 try:
-> 3805     return self._engine.get_loc(casted_key)
   3806 except KeyError as err:

File index.pyx:167, in pandas._libs.index.IndexEngine.get_loc()

File index.pyx:196, in pandas._libs.index.IndexEngine.get_loc()

File pandas\\_libs\\hashtable_class_helper.pxi:2606, in pandas._libs.hashtable.Int64HashTable.get_item()

File pandas\\_libs\\hashtable_class_helper.pxi:2630, in pandas._libs.hashtable.Int64HashTable.get_item()

KeyError: 4605

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
Cell In[36], line 7
      4 for epoch in range(num_epochs):
      5     print(f'Epoch {epoch+1}/{num_epochs}')
----> 7     train_loss, train_acc = train(model, train_loader, optimizer, loss_fn)
      8     valid_loss, valid_acc = evaluate(model, test_loader, loss_fn)
...
   3815     #  InvalidIndexError. Otherwise we fall through and re-raise
   3816     #  the TypeError.
   3817     self._check_indexing_error(key)

KeyError: 4605
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
```

**Cause:** Indicies of dataframe not continuous

**Solution:** Check that the indicies of the dataframe are continous by using ```df.info()``` and re-index using the ```.reset_index()``` method.

```python
df.info()

Out:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 26709 entries, 0 to 26708    <--- verify from here that the values are continous
Data columns (total 2 columns):
 #   Column        Non-Null Count  Dtype 
---  ------        --------------  ----- 
 0   headline      26709 non-null  object
 1   is_sarcastic  26709 non-null  int64 
dtypes: int64(1), object(1)
memory usage: 417.5+ KB
```
## 2. ValueError: Target size (torch.Size([1])) must be the same as input size (torch.Size([])) (or similar; target size not matching input size)

**Example:**
```python
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Cell In[29], line 7
      4 for epoch in range(num_epochs):
      5     print(f'Epoch {epoch+1}/{num_epochs}')
----> 7     train_loss, train_acc = train(model, train_loader, optimizer, loss_fn)
      8     valid_loss, valid_acc = evaluate(model, test_loader, loss_fn)
     10     history.append([train_loss, valid_loss, train_acc, valid_acc])

Cell In[28], line 21
     19 optimizer.zero_grad()
     20 outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids).squeeze()
---> 21 loss = criterion(outputs, targets)
     22 loss.backward()
     23 optimizer.step()

File c:\Users\PanSt\Desktop\4ML\Sarcasm-Detection-using-NLP\.venv\Lib\site-packages\torch\nn\modules\module.py:1532, in Module._wrapped_call_impl(self, *args, **kwargs)
   1530     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1531 else:
-> 1532     return self._call_impl(*args, **kwargs)

File c:\Users\PanSt\Desktop\4ML\Sarcasm-Detection-using-NLP\.venv\Lib\site-packages\torch\nn\modules\module.py:1541, in Module._call_impl(self, *args, **kwargs)
   1536 # If we don't have any hooks, we want to skip the rest of the logic in
   1537 # this function, and just call forward.
   1538 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
...
   3223 if not (target.size() == input.size()):
-> 3224     raise ValueError(f"Target size ({target.size()}) must be the same as input size ({input.size()})")
   3226 return torch.binary_cross_entropy_with_logits(input, target, weight, pos_weight, reduction_enum)

ValueError: Target size (torch.Size([1, 1])) must be the same as input size (torch.Size([]))
```
Training progress bar showing 1064/1065 (last batch)

**Cause:** Last batch of training (or validation) may consist of 1 element in which case it returns a different size than with a batch of 20. For example, instead of returning torch.Size([1, 1]) it returns torch.Size([1]) which causes an issue when applying the ```.squeeze()``` or ```.unsqueeze()``` methods. 

This error can happen at the first batch of training but can also happen right at the last batch if the last batch consists of 1 element. The probability of the latter error happening is 1/batch_size, so for batch_size = 20 it is 5%.

**Solution:** Include if statements checking the dimension of the tensors in using ```.dim()```:

```python
 if data['ids'].dim() == 2: # check if the dimension of the inputs is 2. If so, no unsqeezing is needed
   ids = data['ids'].to(device, dtype = torch.long)
   mask = data['mask'].to(device, dtype = torch.long)
   token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
 elif data['ids'].dim() == 1: # if the values are 1D then add a first dimension: [1, max_len] -> [1, max_len], this will happen when batch_size = 1. Pytorch will return values as 1D
   ids = data['ids'].to(device, dtype = torch.long).unsqueeze(0)
   mask = data['mask'].to(device, dtype = torch.long).unsqueeze(0)
   token_type_ids = data['token_type_ids'].to(device, dtype = torch.long).unsqueeze(0)
 targets = data['targets'].to(device, dtype = torch.float) # targets remain 1D

 optimizer.zero_grad()
 outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
 outputs = outputs.squeeze(0) if outputs.squeeze().dim() == 0 else outputs.squeeze()
 loss = loss_fn(outputs, targets)
 loss.backward()
 optimizer.step()
```
## 3. RuntimeError: The size of tensor a (15000) must match the size of tensor b (512) at non-singleton dimension 1

**Example:**
```python

```

**Cause:** The number of tokens in some sequences exceeds the token window of the encoder. For this particular case, I am using BERT which has a maximum window of 512 tokens. As such sequences that generate more than 512 tokens and are not truncated will cause an error. 

**Solution:** Truncate to 512 tokens (or the respective token window size)

```python
class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.review = dataframe.review
        self.targets = self.data.sentiment
        self.max_len = max_len

    def __len__(self):
        return len(self.review)

    def __getitem__(self, index):
        review = str(self.review[index])
        review = " ".join(review.split())

        inputs = self.tokenizer.encode_plus(
            review,
            truncation='longest_first',
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids='pt'
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }
```
