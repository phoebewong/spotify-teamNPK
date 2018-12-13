import numpy as np

class RecommenderSystem():
    """Represents the scheme for implementing a recommender system."""
    def __init__(self, training_data, *params):
        """Initializes the recommender system.

        Note that training data has to be provided when instantiating.
        Optional parameters are passed to the underlying system.
        """
        raise NotImplementedError

    def train(self, *params):
        """Starts training. Passes optional training parameters to the system."""
        raise NotImplementedError

    def score(self, user_id, hotel_id):
        """Returns a single score for a user-hotel pair.

        If no prediction for the given pair can be made, an exception should be raised.
        """
        raise NotImplementedError

class ALSRecommenderSystem(RecommenderSystem):
    """Provides a biased ALS-based implementation of an implicit recommender system."""
    def __init__(self, training_data, biased, latent_dimension, log_dir=None, confidence=20):
        """Initializes the recommender system.

        Keyword arguments:
        training_data: Data to train on.
        biased: Whether to include user- and item-related biases in the model.
        latent_dimension: Dimension of the latent space.
        log_dir: Optional pointer to directory storing logging information.
        confidence: Confidence value that should be assigned to pairs where interaction
                    was present. Since the data includes single interactions only, simply
                    assigining 1 for non-interactions and this value otherwise suffices.
                    Should be greater than 1.
        """
        self.biased = biased
        self.confidence = confidence
        self.latent_dimension = latent_dimension
        self.U = None
        self.V = None
        self.log_dir = log_dir
        self.C_users, self.P_users, self.C_items, self.P_items, self.mapping_users, self.mapping_hotels = self._build_matrices(training_data, confidence)
        self.user_dim, self.item_dim = self.P_users.shape

    def _build_matrices(self, activity, confidence):
        """Build the initial matrices."""
        distinct_users = len(set(activity['user']))
        distinct_hotels = len(set(activity['hotel']))
        C_users = np.ones(shape=(distinct_users, distinct_hotels))
        P_users = np.zeros(shape=(distinct_users, distinct_hotels))
        C_items = np.ones(shape=(distinct_hotels, distinct_users))
        P_items = np.zeros(shape=(distinct_hotels, distinct_users))

        mapping_users = {}
        mapping_hotels = {}
        user_ct = 0
        hotel_ct = 0

        for index, row in activity.iterrows():
            user, hotel = row
            if not user in mapping_users:
                mapping_users[user] = user_ct
                user_ct += 1
            if not hotel in mapping_hotels:
                mapping_hotels[hotel] = hotel_ct
                hotel_ct += 1
            user_index, hotel_index = mapping_users[user], mapping_hotels[hotel]
            C_users[user_index, hotel_index] = confidence
            P_users[user_index, hotel_index] = 1
            C_items[hotel_index, user_index] = confidence
            P_items[hotel_index, user_index] = 1
        return C_users, P_users, C_items, P_items, mapping_users, mapping_hotels

    def save(self, directory):
        """Saves current matrices to the given directory."""
        np.save(os.path.join(directory, 'U.npy'), self.U)
        np.save(os.path.join(directory, 'V.npy'), self.V)
        np.save(os.path.join(directory, 'training_data.npy'), self.training_data)
        np.save(os.path.join(directory, 'params.npy'), np.array([self.confidence]))
        if self.biased:
            np.save(os.path.join(directory, 'user_biases.npy'), self.user_biases)
            np.save(os.path.join(directory, 'item_biases.npy'), self.item_biases)

    def load(self, directory):
        """Loads matrices from the given directory."""
        self.U = np.load(os.path.join(directory, 'U.npy'))
        self.V = np.load(os.path.join(directory, 'V.npy'))
        self.training_data = np.load(os.path.join(directory, 'training_data.npy'))
        self.confidence = np.load(os.path.join(directory, 'params.npy')).flatten()
        if self.biased:
            self.user_biases = np.load(os.path.join(directory, 'user_biases.npy'))
            self.item_biases = np.load(os.path.join(directory, 'item_biases.npy'))

        self.C_users, self.P_users, self.C_items, self.P_items, self.mapping_users, self.mapping_hotels = self._build_matrices(self.training_data, self.confidence)
        self.user_dim, self.item_dim = self.P_users.shape

    def _single_step(self, lbd):
        """Executes a single optimization step using (biased) ALS, with lbd as regularization factor."""
        C_users, P_users, C_items, P_items, mapping_users, mapping_hotels = self.C_users, self.P_users, self.C_items, self.P_items, self.mapping_users, self.mapping_hotels
        biased = self.biased

        # Update U.
        if biased: # Expand matrices to account for biases.
            U_exp = np.hstack((self.user_biases.reshape(-1,1), self.U))
            V_exp = np.hstack((np.ones_like(self.item_biases).reshape(-1,1), self.V))
            kdim = self.latent_dimension + 1
        else: # We work with copies here to make it safer to abort within updates.
            U_exp = self.U.copy()
            V_exp = self.V.copy()
            kdim = self.latent_dimension
        Vt = np.dot(np.transpose(V_exp), V_exp)
        for user_index in tqdm(range(self.user_dim)):
            C = np.diag(C_users[user_index])
            d = np.dot(C, P_users[user_index] - (0 if not biased else self.item_biases))
            val = np.dot(np.linalg.inv(Vt + np.dot(np.dot(V_exp.T, C - np.eye(self.item_dim)), V_exp) + lbd*np.eye(kdim)), np.transpose(V_exp))
            U_exp[user_index] = np.dot(val, d)
        if biased:
            self.user_biases = U_exp[:,0]
            self.U = U_exp[:,1:]
        else:
            self.U = U_exp

        # Update V.
        if biased:
            U_exp = np.hstack((np.ones_like(self.user_biases).reshape(-1,1), self.U))
            V_exp = np.hstack((self.item_biases.reshape(-1,1), self.V))
        else: # We work with copies here to make it safer to abort within updates.
            U_exp = self.U.copy()
            V_exp = self.V.copy()

        Ut = np.dot(np.transpose(U_exp), U_exp)
        for item_index in tqdm(range(self.item_dim)):
            C = np.diag(C_items[item_index])
            d = np.dot(C, P_items[item_index] - (0 if not biased else self.user_biases))
            val = np.dot(np.linalg.inv(Ut + np.dot(np.dot(U_exp.T, C-np.eye(self.user_dim)), U_exp) + lbd*np.eye(kdim)), np.transpose(U_exp))
            V_exp[item_index] = np.dot(val, d)
        if biased:
            self.item_biases = V_exp[:, 0]
            self.V = V_exp[:,1:]
        else:
            self.V = V_exp

    def compute_loss(self, lbd):
        """Computes loss value on the training data.

        Returns a tuple of total loss and prediction loss (excluding regularization loss).
        """
        C_users, P_users, C_items, P_items, mapping_users, mapping_hotels = self.C_users, self.P_users, self.C_items, self.P_items, self.mapping_users, self.mapping_hotels
        main_loss = 0
        # Main loss term.
        for user_index in range(self.user_dim):
            for item_index in range(self.item_dim):
                pred = np.dot(self.U[user_index].T, self.V[item_index])
                if self.biased:
                    pred += self.user_biases[user_index] + self.item_biases[item_index]
                loss = self.C_users[user_index, item_index] * (P_users[user_index, item_index]-pred)**2
                main_loss += loss

        # Regularization term.
        reg_loss = 0
        if lbd > 0:
            for user_index in range(self.user_dim):
                reg_loss += np.sum(self.U[user_index]**2) + (0 if not self.biased else self.user_biases[user_index]**2)
            for item_index in range(self.item_dim):
                reg_loss += np.sum(self.V[item_index]**2) + (0 if not self.biased else self.item_biases[item_index]**2)
            reg_loss *= lbd
        return main_loss + reg_loss, main_loss

    def train(self, lbd, iterations=20, verbose=True):
        """
        Trains the recommendation system.

        Keyword arguments:
        lbd: Regularization factor.
        iterations: Number of iterations to run ALS.
        verbose: Whether to plot and output training loss.
        """
        if self.U is None or self.V is None:
            self.U = np.random.normal(size=(self.user_dim, self.latent_dimension))
            self.V = np.random.normal(size=(self.item_dim, self.latent_dimension))
            self.user_biases = np.zeros(self.user_dim)
            self.item_biases = np.zeros(self.item_dim)
            self.history_losses = []
            self.history_main_losses = []
            self.history_avg_score = []
            self.history_avg_rank = []

        it = 0
        while(it < iterations):
            self._single_step(lbd)
            loss, main_loss = self.compute_loss(lbd)
            self.history_losses.append(loss)
            self.history_main_losses.append(main_loss)

            if verbose:
                clear_output(wait=True)
                print('LOSS:', loss, 'MAIN LOSS:', main_loss)

                plt.figure(figsize=(5,5))
                plt.title('training loss (lower is better)')
                plt.plot(range(len(self.history_losses)), self.history_losses)
                plt.plot(range(len(self.history_main_losses)), self.history_main_losses, color='orange')
                plt.plot(range(len(self.history_main_losses)), np.array(self.history_losses) - np.array(self.history_main_losses), color='green')
                plt.legend(['total loss', 'data loss', 'regularizing loss'])
                if self.log_dir is not None:
                    plt.savefig(os.path.join(self.log_dir, 'log.png'), bbox_inches='tight', format='png')
                plt.show()
            it += 1

    def reset(self):
        """Resets the recommendation system's internal state."""
        self.U = None
        self.V = None

    def score(self, user_id, hotel_id):
        """Returns the scoring of hotel_id for user_id."""
        if self.U is None or self.V is None:
            raise ValueError('system has to be trained first')
        if user_id not in self.mapping_users:
            raise ValueError('user unknown')
        if hotel_id not in self.mapping_hotels:
            raise ValueError('hotel unknown')

        user_index = self.mapping_users[user_id]
        hotel_index = self.mapping_hotels[hotel_id]
        pred = np.dot(self.U[user_index], self.V[hotel_index])
        if self.biased: # Include applicable biases.
            pred += self.user_biases[user_index] + self.item_biases[hotel_index]
        return pred
