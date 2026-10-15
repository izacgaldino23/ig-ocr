import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import torch
from torchvision import transforms
from model import SimpleNeuralNetwork
from args import get_args

args = get_args()

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconhecedor de Dígitos MNIST")
        
        # 1. Carregando o modelo treinado
        self.model = SimpleNeuralNetwork()
        self.model.load_state_dict(torch.load(f'data/models/{args.input}'))
        self.model.eval()

        # 2. Criando a área de desenho (Canvas)
        self.canvas_size = 280
        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size, bg='black')
        self.canvas.pack(pady=10)
        
        # Vinculando o evento de arrastar o mouse para desenhar
        self.canvas.bind("<B1-Motion>", self.draw)

        # 3. Imagem auxiliar do Pillow para guardar o desenho em alta resolução
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.draw_tool = ImageDraw.Draw(self.image)

        # 4. Botões de Ação
        self.btn_predict = tk.Button(root, text="Prever Dígito", command=self.predict)
        self.btn_predict.pack(side=tk.LEFT, padx=20, pady=10)

        self.btn_clear = tk.Button(root, text="Limpar", command=self.clear_canvas)
        self.btn_clear.pack(side=tk.RIGHT, padx=20, pady=10)

        # 5. Rótulo para exibir o resultado
        self.label_result = tk.Label(root, text="Desenhe um número acima", font=("Arial", 14))
        self.label_result.pack(pady=10)

    def draw(self, event):
        # Desenha um círculo branco onde o mouse passa
        r = 8
        x, y = event.x, event.y
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='white', outline='white')
        self.draw_tool.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.draw_tool = ImageDraw.Draw(self.image)
        self.label_result.config(text="Desenhe um número acima")

    def predict(self):
        # Redimensiona para 28x28 pixels para corresponder ao MNIST
        img_resized = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Transforma em tensor e normaliza
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        tensor_image = transform(img_resized).unsqueeze(0)

        # Faz a predição com o modelo
        with torch.no_grad():
            output = self.model(tensor_image)
            _, predicted = torch.max(output, 1)

        self.label_result.config(text=f"Previsão da Rede: {predicted.item()}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()