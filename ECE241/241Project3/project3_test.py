from project3p import *

input = rf'/Users/aaditya/Documents/UMASS/ECE241/Project 3/train.csv'

model1 = Model(input)
model2 = Model(input)

# ==============================
''' INITIALIZING STATEMENTS '''
# ==============================
first_ten_items = model1.first_ten_items()
size = model1.size()
mean_price = model1.mean_price()
min_and_max_price = model1.min_and_max_price()
std_dev = model1.standard_deviation()
# hist_price = model1.hist_price()
# scatter_plot = model1.pair_wise_scatter_plot()


prediciton = model1.pred()
# prediciton2 = model2.pred()


loss = model1.loss()
# loss2 = model2.loss()

gradient = model1.gradient()
# gradient2 = model2.gradient()

update = model1.update(0.2)
update2 = model2.update(0.2)


# ========================
''' PRINT STATEMENTS '''
# ========================
print(first_ten_items)
print(size)
print(mean_price)
print(min_and_max_price)
print(std_dev)
print(prediciton)
print("========================================================================================================")
print(loss)
print("========================================================================================================")
print(gradient)
print("========================================================================================================")
#print(update)

x = 10**-9
y = 10**-9.6

run1 = model1.TrainModel(x)
run2 = model2.TrainModel(y)

iteration_run1 = range(len(run1))
iteration_run2 = range(len(run2))

plt.plot(iteration_run1,run1,label = x)
plt.plot(iteration_run2,run2,label = y)

print(sum(run1),sum(run2))
plt.title(label='ECE 241')
plt.legend()
plt.show()

