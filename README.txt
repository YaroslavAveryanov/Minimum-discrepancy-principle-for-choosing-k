
Requirements:
matplotlib>=3.1.1
numpy>=1.18.5
scikit-learn>=0.21.3
seaborn>=0.9
scipy>=1.4.1
pandas>=0.25.1

In order to reproduce all experiments from the paper
"Minimum discrepancy principle strategy for choosing k in k-NN regression":

-- Figure 2(a):
    -python main.py
    -Type 0 if artificial data, 1 if real data:
     '0'
     Type 0 if smooth function, 1 if sinus function:
     '0'
     Type the number of repetitions to perform:
     '1000'

-- Figure 2(b):
    -python main.py
    -Type 0 if artificial data, 1 if real data:
     '0'
     Type 0 if smooth function, 1 if sinus function:
     '1'
     Type the value of noise (std):
     '0.1'
     Type the number of repetitions to perform:
     '1000'

-- Figures 4(a),(b):
   -python main.py
   -Type 0 if artificial data, 1 if real data:
    '1'
    Type the data you want to choose:
    Type 0 if Boston House Prices data,
    1 if Diabetes data,
    2 if California House Prices data,
    3 if Power Plants data:
    '0'

-- Figures 4(c),(d):
  -python main.py
  -Type 0 if artificial data, 1 if real data:
   '1'
   Type the data you want to choose:
   Type 0 if Boston House Prices data,
   1 if Diabetes data,
   2 if California House Prices data,
   3 if Power Plants data:
   '1'

-- Figures 5(a),(b):
    -python main.py
    -Type 0 if artificial data, 1 if real data:
     '1'
     Type the data you want to choose:
     Type 0 if Boston House Prices data,
     1 if Diabetes data,
     2 if California House Prices data,
     3 if Power Plants data:
     '2'

-- Figures 5(c),(d):
    -python main.py
    -Type 0 if artificial data, 1 if real data:
     '1'
     Type the data you want to choose:
     Type 0 if Boston House Prices data,
     1 if Diabetes data,
     2 if California House Prices data,
     3 if Power Plants data:
     '3'
