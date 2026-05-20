import tkinter as tk
from tkinter import scrolledtext, messagebox
import subprocess
import threading

class SparqlGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SPARQL Assistant")
        self.root.geometry("800x650")
        self.root.resizable(True, True)
        self.model_name = "qwen-sparql999.gguf"
        tk.Label(root, text="SPARQL Assistant", font=("Arial", 16, "bold")).pack(pady=10)
        tk.Label(root, text="Генерация SPARQL-запросов по вопросам на русском языке", font=("Arial", 10)).pack()
        self.status_model = tk.Label(root, text=f"Модель: {self.model_name}", fg="green", font=("Arial", 9))
        self.status_model.pack(pady=5)
        
        tk.Label(root, text="Вопрос:", anchor="w", font=("Arial", 11)).pack(pady=(15,5), padx=10, anchor="w")
        self.question_text = tk.Text(root, height=5, font=("Arial", 11), wrap="word")
        self.question_text.pack(fill="x", padx=10)
        
        self.generate_btn = tk.Button(root, text="Сгенерировать SPARQL", font=("Arial", 11),
                                      command=self.generate, bg="#3498db", fg="white")
        self.generate_btn.pack(pady=10)
        
        self.status_label = tk.Label(root, text="", fg="gray")
        self.status_label.pack()
        
        result_frame = tk.Frame(root)
        result_frame.pack(fill="both", expand=True, padx=10, pady=(10,0))
        
        tk.Label(result_frame, text="SPARQL запрос:", anchor="w", font=("Arial", 11)).pack(anchor="w")
        
        self.copy_btn = tk.Button(result_frame, text="Копировать", font=("Arial", 9),
                                  command=self.copy_to_clipboard, bg="#f0f0f0")
        self.copy_btn.pack(anchor="e", pady=(0,5))
        
        self.result_text = scrolledtext.ScrolledText(result_frame, height=12, font=("Courier", 10), wrap="word")
        self.result_text.pack(fill="both", expand=True)
        
        self.setup_context_menu()
        
    
    def setup_context_menu(self):
        """Создает контекстное меню для копирования текста"""
        self.context_menu = tk.Menu(self.root, tearoff=0)
        self.context_menu.add_command(label="Копировать", command=self.copy_to_clipboard)
        self.context_menu.add_command(label="Вырезать", command=self.cut_text)
        self.context_menu.add_command(label="Выделить всё", command=self.select_all)
        
        self.result_text.bind("<Button-3>", self.show_context_menu)
        self.question_text.bind("<Button-3>", self.show_context_menu)
    
    def show_context_menu(self, event):
        """Показывает контекстное меню"""
        try:
            self.context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.context_menu.grab_release()
    
    def copy_to_clipboard(self):
        """Копирует выделенный текст или весь текст результата"""
        try:
            selected = self.result_text.get(tk.SEL_FIRST, tk.SEL_LAST)
            self.root.clipboard_clear()
            self.root.clipboard_append(selected)
            self.status_label.config(text="Скопировано", fg="green")
            self.root.after(2000, lambda: self.status_label.config(text="", fg="gray"))
        except tk.TclError:
            all_text = self.result_text.get("1.0", tk.END).strip()
            if all_text:
                self.root.clipboard_clear()
                self.root.clipboard_append(all_text)
                self.status_label.config(text="Скопировано все", fg="green")
                self.root.after(2000, lambda: self.status_label.config(text="", fg="gray"))
            else:
                self.status_label.config(text="Нечего копировать", fg="red")
                self.root.after(2000, lambda: self.status_label.config(text="", fg="gray"))
    
    def cut_text(self):
        """Вырезает выделенный текст"""
        try:
            selected = self.result_text.get(tk.SEL_FIRST, tk.SEL_LAST)
            self.copy_to_clipboard()
            self.result_text.delete(tk.SEL_FIRST, tk.SEL_LAST)
            self.status_label.config(text="Вырезано", fg="green")
            self.root.after(2000, lambda: self.status_label.config(text="", fg="gray"))
        except tk.TclError:
            pass
    
    def select_all(self):
        """Выделяет весь текст"""
        self.result_text.tag_add(tk.SEL, "1.0", tk.END)
        self.result_text.mark_set(tk.INSERT, "1.0")
        self.result_text.see(tk.INSERT)
    
    def set_question(self, question):
        self.question_text.delete("1.0", tk.END)
        self.question_text.insert("1.0", question)
    
    def generate(self):
        question = self.question_text.get("1.0", tk.END).strip()
        if not question:
            messagebox.showwarning("Внимание", "Пожалуйста, введите вопрос")
            return
        
        self.generate_btn.config(state="disabled")
        self.copy_btn.config(state="disabled")
        self.status_label.config(text="Происходит генерация запроса, пожалуйста, подождите...")
        self.result_text.delete("1.0", tk.END)
        
        thread = threading.Thread(target=self._call_ollama, args=(question,))
        thread.start()
    
    def _call_ollama(self, question):
        try:
            result = subprocess.run(
                ['ollama', 'run', self.model_name, question],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                output = result.stdout.strip()
                self.root.after(0, self._update_result, output)
            else:
                error_msg = result.stderr.strip()
                self.root.after(0, self._update_result, f"Ошибка:\n{error_msg}")
                
        except subprocess.TimeoutExpired:
            self.root.after(0, self._update_result, "Ошибка, превышено время ожидания в 60 секунд")
        except FileNotFoundError:
            self.root.after(0, self._update_result, "Ошибка, Ollama не найден. Прошу, убедитесь, что Ollama установлен и запущен")
        except Exception as e:
            self.root.after(0, self._update_result, f"Ошибка, {e}")
    
    def _update_result(self, text):
        self.result_text.insert("1.0", text)
        self.status_label.config(text="Готово")
        self.generate_btn.config(state="normal")
        self.copy_btn.config(state="normal")

if __name__ == "__main__":
    root = tk.Tk()
    app = SparqlGUI(root)
    root.mainloop()
