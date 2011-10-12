import numpy


file_name = 'T-61_3050_data.txt'
row_numbers = numpy.loadtxt(file_name, comments='%', usecols=[0])
data = numpy.loadtxt(file_name, comments='%', usecols=[1:], dtype=bool, 
    converters={1 : lambda x: (x if x != 'NaN' else False)})

#data[:, 2:] = (data[:, 2:] == 1)

print data[:, 2:]


def impurity(data, col):

  imp = -(Nm[0] / Nm
	  + )

  return imp


def build_tree(data, impurity_treshold):

  rows = numpy.size(data, 0)
  columns = numpy.size(data, 1);
  best_impurity = inf

  for col in range(0, columns):
    
    t = impurity(data, col)
    if t < best_impurity:
      best_impurity = t
      best_column = col

  Lidx = data[:, best_column] == 1

  if impurity < impurity_treshold:
    return n = size(Lidx) / rows

  Ridx = data[:, best_column] == 0
  lesserdata = numpy.delete(data, best_column, 1)
  L = build_tree(lesserdata[Lidx])
  R = build_tree(lesserdata[Ridx])

  return [L, R]
