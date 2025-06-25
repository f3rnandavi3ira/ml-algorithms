import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import random

# 1. Carregar o conjunto de dados de dígitos manuscritos
digits = datasets.load_digits()

# 2. Visualizar alguns exemplos de "respostas de alunos"
plt.figure(figsize=(8, 2))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(digits.images[i], cmap='gray')
    plt.axis('off')
    plt.title(str(digits.target[i]))
plt.suptitle("Exemplos de respostas escritas à mão por alunos")
plt.show()

# 3. Preparar os dados
X = digits.data  # imagens achatadas: 8x8 -> 64 features
y = digits.target  # rótulos: 0 a 9

# 4. Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Dividir entre treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=5)

# 6. Criar e treinar o modelo SVM com kernel RBF
svm_clf = SVC(kernel='rbf', gamma=0.05, C=1.0)
svm_clf.fit(X_train, y_train)

# 7. Fazer previsões
y_pred = svm_clf.predict(X_test)

# 8. Avaliação do modelo (simulando correção automática)
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# 9. Simular a correção automática de 10 "respostas de alunos"
plt.figure(figsize=(12, 4))
for i in range(10):
    index = random.randint(0, len(X_test)-1)
    img = X_test[index].reshape(8, 8)
    true_label = y_test[index]
    pred_label = y_pred[index]
    
    plt.subplot(2, 5, i + 1)
    plt.imshow(img, cmap='gray')
    color = 'green' if pred_label == true_label else 'red'
    plt.title(f"Aluno: {true_label}\nModelo: {pred_label}", color=color)
    plt.axis('off')

plt.suptitle("Simulação de correção automática (verde = certo, vermelho = errado)")
plt.show()
