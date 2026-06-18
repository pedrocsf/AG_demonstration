import numpy as np
import random
import pickle
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

from preparacao_de_dados import AlgoritmoGenetico_preparacao_de_dados, calcular_aptidao

# ==========================================
# 1. CLASSE DO CLASSIFICADOR SERIALIZÁVEL
# ==========================================
class AlgoritmoGenetico_classificador:
    """Modelo final que encapsula a regra gerada, os metadados de imputação e o scaler."""
    def __init__(self, s, t, d, k, w, medians, scaler, feature_names):
        self.s = s
        self.t = t
        self.d = d
        self.k = k
        self.w = w
        self.medians = medians
        self.scaler = scaler
        self.feature_names = feature_names

    def preprocess(self, x_raw):
        x_imp = np.array(x_raw, dtype=float).copy()
        if x_imp.ndim == 1:
            x_imp = x_imp.reshape(1, -1)
        cols_to_impute = [1, 2, 3, 4, 5]
        for c in cols_to_impute:
            x_imp[:, c] = np.where((x_imp[:, c] == 0) | np.isnan(x_imp[:, c]), self.medians[c], x_imp[:, c])
        return self.scaler.transform(x_imp)

    def predict(self, x_raw):
        x_scaled = self.preprocess(x_raw)
        results = []
        for x in x_scaled:
            met_weight = 0.0   # soma dos pesos das regras satisfeitas
            total_weight = 0.0  # soma dos pesos de todas as regras ativas
            for i in range(len(self.s)):
                if self.s[i] == 1:
                    total_weight += self.w[i]
                    if self.d[i] == 1 and x[i] > self.t[i]:
                        met_weight += self.w[i]
                    elif self.d[i] == 0 and x[i] < self.t[i]:
                        met_weight += self.w[i]
            # Inferência normalizada em [0, 1]: fração da evidência ponderada satisfeita
            score = (met_weight / total_weight) if total_weight > 0 else 0.0
            pred = 1 if score >= self.k else 0
            results.append((pred, score, self.k))
        return results[0] if np.array(x_raw).ndim == 1 else results

# ==========================================
# 2. REPRESENTAÇÃO CROMOSSÔMICA DO INDIVÍDUO
# ==========================================
class Individual:
    def __init__(self, n_features, t_bounds=None):
        self.n_features = n_features
        self.s = np.random.randint(2, size=n_features)
        self.d = np.random.randint(2, size=n_features)
        if t_bounds is not None:
            self.t = np.array([random.uniform(b[0], b[1]) for b in t_bounds])
        else:
            self.t = np.zeros(n_features)

        # Peso real de cada regra em [0, 1]
        self.w = np.random.uniform(0.0, 1.0, size=n_features)
        # Limiar mínimo de inferência (fração ponderada satisfeita) em [0, 1]
        self.k = random.uniform(0.0, 1.0)
        self.fitness = -1.0

    def copy(self):
        ind = Individual(self.n_features)
        ind.s = self.s.copy()
        ind.d = self.d.copy()
        ind.t = self.t.copy()
        ind.w = self.w.copy()
        ind.k = self.k
        ind.fitness = self.fitness
        return ind

# ==========================================
# 3. OPERADORES GENÉTICOS
# ==========================================
def crossover(p1, p2):
    if random.random() < 0.8:
        c1, c2 = p1.copy(), p2.copy()
        # Cruzamento de 1 Ponto para os genes binários (s, d)
        bin1 = list(p1.s) + list(p1.d)
        bin2 = list(p2.s) + list(p2.d)
        cp = random.randint(1, 15)
        new_bin1 = bin1[:cp] + bin2[cp:]
        new_bin2 = bin2[:cp] + bin1[cp:]

        c1.s, c1.d = np.array(new_bin1[:8]), np.array(new_bin1[8:16])
        c2.s, c2.d = np.array(new_bin2[:8]), np.array(new_bin2[8:16])

        # Cruzamento BLX-alpha para valores reais (Limiares t e Pesos w)
        alpha = 0.5
        for i in range(p1.n_features):
            # Limiar t
            c_min, c_max = min(p1.t[i], p2.t[i]), max(p1.t[i], p2.t[i])
            diff = c_max - c_min
            c1.t[i] = random.uniform(c_min - alpha * diff, c_max + alpha * diff)
            c2.t[i] = random.uniform(c_min - alpha * diff, c_max + alpha * diff)
            # Peso w (mantido em [0, 1])
            w_min, w_max = min(p1.w[i], p2.w[i]), max(p1.w[i], p2.w[i])
            w_diff = w_max - w_min
            c1.w[i] = np.clip(random.uniform(w_min - alpha * w_diff, w_max + alpha * w_diff), 0.0, 1.0)
            c2.w[i] = np.clip(random.uniform(w_min - alpha * w_diff, w_max + alpha * w_diff), 0.0, 1.0)

        # Cruzamento BLX-alpha para o limiar de inferência k (mantido em [0, 1])
        k_min, k_max = min(p1.k, p2.k), max(p1.k, p2.k)
        k_diff = k_max - k_min
        c1.k = float(np.clip(random.uniform(k_min - alpha * k_diff, k_max + alpha * k_diff), 0.0, 1.0))
        c2.k = float(np.clip(random.uniform(k_min - alpha * k_diff, k_max + alpha * k_diff), 0.0, 1.0))
        return c1, c2
    return p1.copy(), p2.copy()

def mutate(ind, stds_train):
    mutation_rate = 0.02
    for i in range(ind.n_features):
        if random.random() < mutation_rate: ind.s[i] = 1 - ind.s[i]
        if random.random() < mutation_rate: ind.d[i] = 1 - ind.d[i]
        if random.random() < mutation_rate: ind.t[i] += random.gauss(0, stds_train[i])
        if random.random() < mutation_rate:
            ind.w[i] = np.clip(ind.w[i] + random.gauss(0, 0.1), 0.0, 1.0)

    if random.random() < mutation_rate:
        ind.k = float(np.clip(ind.k + random.gauss(0, 0.1), 0.0, 1.0))

# ==========================================
# 4. EXECUÇÃO PRINCIPAL DA EVOLUÇÃO
# ==========================================
def main():
    # Extrai as configurações diretamente do módulo de preparação (Script 1)
    X_train_scaled, X_test_raw, y_train, y_test, cv_folds, t_bounds, stds_train, medians, scaler, feature_names = AlgoritmoGenetico_preparacao_de_dados()

    pop_size = 200
    max_gens = 500
    patience = 50
    
    pop = [Individual(8, t_bounds) for _ in range(pop_size)]
    best_fitness_global = -1.0
    best_ind_global = None
    gens_without_improvement = 0

    print("Iniciando Evolução (Numpy Puro)...")
    for gen in range(max_gens):
        for ind in pop:
            if ind.fitness == -1.0:
                # Utiliza a função objetivo exportada pelo Script 1
                ind.fitness = calcular_aptidao(ind.s, ind.t, ind.d, ind.k, ind.w, X_train_scaled, y_train, cv_folds)

        pop.sort(key=lambda x: x.fitness, reverse=True)
        
        if pop[0].fitness > best_fitness_global:
            best_fitness_global = pop[0].fitness
            best_ind_global = pop[0].copy()
            gens_without_improvement = 0
            print(f"Gen {gen:3d} | F1-Macro CV: {pop[0].fitness:.4f} (k={pop[0].k:.3f}) *** novo melhor ***")
        else:
            gens_without_improvement += 1
            print(f"Gen {gen:3d} | F1-Macro CV: {pop[0].fitness:.4f} (k={pop[0].k:.3f}) | sem melhora: {gens_without_improvement}/{patience}")

        if gens_without_improvement >= patience:
            print(f"\nCritério de parada atingido: {patience} gerações sem melhora (Geração {gen}).")
            break

        new_pop = [ind.copy() for ind in pop[:10]] # Elitismo
        
        while len(new_pop) < pop_size:
            t1 = max(random.sample(pop, 5), key=lambda x: x.fitness)
            t2 = max(random.sample(pop, 5), key=lambda x: x.fitness)
            c1, c2 = crossover(t1, t2)
            mutate(c1, stds_train)
            mutate(c2, stds_train)
            c1.fitness, c2.fitness = -1.0, -1.0
            new_pop.extend([c1, c2])
            
        pop = new_pop[:pop_size]

    print("\n" + "="*50 + "\nRESULTADOS FINAIS NO CONJUNTO DE TESTE\n" + "="*50)
    t_orig = scaler.inverse_transform([best_ind_global.t])[0]
    
    print(f"\n[REGRA DE DECISÃO EXPLÍCITA - Melhor Indivíduo]")
    for i in range(8):
        if best_ind_global.s[i] == 1:
            print(f"SE {feature_names[i]} {'>' if best_ind_global.d[i] == 1 else '<'} {t_orig[i]:.2f} (peso={best_ind_global.w[i]:.3f}) E")
    print(f"ENTÃO Diabetes (Inferência ponderada normalizada >= {best_ind_global.k:.3f})")

    model = AlgoritmoGenetico_classificador(best_ind_global.s, best_ind_global.t, best_ind_global.d, best_ind_global.k, best_ind_global.w, medians, scaler, feature_names)
    predictions = model.predict(X_test_raw)
    y_pred = [p[0] for p in predictions]

    print(f"Acurácia (Teste): {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()