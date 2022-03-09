# crm

Neural networks that are explainable by design

## Usage

```console
$ python3 main.py -h
usage: main.py [-h] -f FILE -o OUTPUT -n NUM_EPOCHS [-s SAVED_MODEL] [-e] [-v] [-g]

CRM; Example: python3 main.py -f inp.file -o out.file -n 20 -s saved_model.pt -e -v

optional arguments:
  -h, --help            show this help message and exit
  -f FILE, --file FILE  input file
  -o OUTPUT, --output OUTPUT
                        output file
  -n NUM_EPOCHS, --num-epochs NUM_EPOCHS
                        number of epochs
  -s SAVED_MODEL, --saved-model SAVED_MODEL
                        location of saved model
  -e, --explain         get explanations for predictions
  -v, --verbose         get verbose outputs
  -g, --gpu             run model on gpu

```
