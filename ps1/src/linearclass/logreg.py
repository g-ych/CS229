import numpy as np
import util

def sigmoid(x):
	return 1 /(1 + np.exp(-x))

def main(train_path, valid_path, save_path):
	"""Problem: Logistic regression with Newton's Method.

	Args:
		train_path: Path to CSV file containing dataset for training.
		valid_path: Path to CSV file containing dataset for validation.
		save_path: Path to save predicted probabilities using np.savetxt().
	"""
	x_train, y_train = util.load_dataset(train_path, add_intercept=True)


	# *** START CODE HERE ***
	# Train a logistic regression classifier
	model = LogisticRegression()
	model.fit(x_train, y_train)
	# Plot decision boundary on top of validation set set
	x_val, y_val = util.load_dataset(valid_path, add_intercept=True)
	y_pred = model.predict(x_val)
	util.plot(x_val, y_val, model.theta, '{}.png'.format(save_path))
	# Use np.savetxt to save predictions on eval set to save_path
	np.savetxt(save_path, y_pred)
	# *** END CODE HERE ***
class LogisticRegression:
	"""Logistic regression with Newton's Method as the solver.

	Example usage:
		> clf = LogisticRegression()
		> clf.fit(x_train, y_train)
		> clf.predict(x_eval)
	"""
	def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
				 theta_0=None, verbose=True):
		"""
		Args:
			step_size: Step size for iterative solvers only.
			max_iter: Maximum number of iterations for the solver.
			eps: Threshold for determining convergence.
			theta_0: Initial guess for theta. If None, use the zero vector.
			verbose: Print loss values during training.
		"""
		self.theta = theta_0
		self.step_size = step_size
		self.max_iter = max_iter
		self.eps = eps
		self.verbose = verbose

	def fit(self, x, y):
		"""Run Newton's Method to minimize J(theta) for logistic regression.

		Args:
			x: Training example inputs. Shape (n_examples, dim).
			y: Training example labels. Shape (n_examples,).
		"""
		# *** START CODE HERE ***
		n, d = x.shape
		# initialize theta
		if self.theta is None:
			self.theta = np.zeros(d)
		# optimize theta
		step = 0
		while True:
			theta = self.theta
			x_theta = x.dot(theta)
			Nabla = - (1 / n) * (y - sigmoid(x_theta)).dot(x)
			H = (1 / n) * sigmoid(x_theta).dot((1 - sigmoid(x_theta))) * (x.T).dot(x)
			H_inv = np.linalg.inv(H)
			self.theta = theta - H_inv.dot(Nabla)
			step = step + 1
			if np.linalg.norm(self.theta - theta, ord=1) < self.eps or step > self.max_iter:
				break
		# *** END CODE HERE ***

	def predict(self, x):
		"""Return predicted probabilities given new inputs x.

		Args:
			x: Inputs of shape (n_examples, dim).

		Returns:
			Outputs of shape (n_examples,).
		"""
		# *** START CODE HERE ***
		# compute probability
		pred = sigmoid(x.dot(self.theta))
		return pred
		# *** END CODE HERE ***

if __name__ == '__main__':
	main(train_path='ds1_train.csv',
		 valid_path='ds1_valid.csv',
		 save_path='logreg_pred_1.txt')

	main(train_path='ds2_train.csv',
		 valid_path='ds2_valid.csv',
		 save_path='logreg_pred_2.txt')
