import { useState, useRef, useEffect, type ChangeEvent } from "react";
import { m, AnimatePresence } from "framer-motion";
import axios from "axios";

interface Category {
  name: string;
  probability: number;
}

type Message = {
  id: string;
  text: string;
  isUser: boolean;
  isLoading?: boolean;
  categories?: Category[];
  error?: string;
  fileName?: string;
};

export default function App() {
  const API_URL = import.meta.env.VITE_API_URL;
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const copyToClipboard = (text: string, messageId: string) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopiedId(messageId);
      setTimeout(() => setCopiedId(null), 2000);
    });
  };

  const formatResults = (categories: Category[] = []) => {
    return categories
      .map((cat) => `${cat.name}: ${cat.probability.toFixed(1)}%`)
      .join("\n");
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleSendMessage = async () => {
    const text = inputValue.trim();
    if ((!text && !file) || isLoading) return;

    const userMessageId = `user-${Date.now()}`;
    const loadingMessageId = `loading-${Date.now()}`;

    setMessages((prev) => [
      ...prev,
      {
        id: userMessageId,
        text: file ? `Файл: ${file.name}` : text,
        isUser: true,
        isLoading: false,
        fileName: file?.name,
      },
    ]);

    setInputValue("");
    setFile(null);
    setIsLoading(true);

    setMessages((prev) => [
      ...prev,
      {
        id: loadingMessageId,
        text: "",
        isUser: false,
        isLoading: true,
      },
    ]);

    try {
      let response;

      if (file) {
        const formData = new FormData();
        formData.append("file", file);
        setIsUploading(true);

        response = await axios.post<{ categories: Category[] }>(
          `${API_URL}/upload`,
          formData,
          {
            headers: {
              "Content-Type": "multipart/form-data",
            },
            timeout: 30000,
          },
        );
      } else {
        response = await axios.post<{ categories: Category[] }>(
          `${API_URL}/analyze`,
          { text },
          {
            headers: {
              "Content-Type": "application/json",
            },
            timeout: 10000,
          },
        );
      }

      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === loadingMessageId
            ? {
                ...msg,
                isLoading: false,
                categories: response.data.categories,
              }
            : msg,
        ),
      );
    } catch (error) {
      let errorMessage = "Неизвестная ошибка";

      if (axios.isAxiosError(error)) {
        if (error.response) {
          errorMessage = `Ошибка ${error.response.status}: ${
            error.response.data.message || error.message
          }`;
        } else if (error.request) {
          errorMessage = "Нет ответа от сервера. Проверьте подключение.";
        } else {
          errorMessage = `Ошибка запроса: ${error.message}`;
        }
      } else if (error instanceof Error) {
        errorMessage = error.message;
      }

      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === loadingMessageId
            ? {
                ...msg,
                isLoading: false,
                error: errorMessage,
              }
            : msg,
        ),
      );
    } finally {
      setIsLoading(false);
      setIsUploading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="flex min-h-screen w-screen bg-gray-50">
      {/* Левая рамка */}
      <div className="w-16 bg-indigo-50 md:w-32 lg:w-48"></div>

      {/* Основное содержимое */}
      <div className="flex flex-1 flex-col">
        <header className="sticky top-0 z-100 flex bg-white shadow-sm">
          <div className="mx-auto flex max-w-4xl justify-center px-6 py-6">
            <m.h1
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.9 }}
              className="text-xl font-bold text-indigo-600 md:text-3xl"
            >
              Анализатор ценностей
            </m.h1>
          </div>
        </header>

        <main className="flex flex-1 flex-col overflow-hidden overflow-x-hidden">
          <div className="flex-1 space-y-4 overflow-y-auto p-4">
            <AnimatePresence>
              {messages.length === 0 ? (
                <m.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex h-full flex-col items-center justify-center p-8 text-center"
                >
                  <m.div
                    animate={{
                      scale: [1, 1.1, 1],
                      rotate: [0, 5, -5, 0],
                    }}
                    transition={{
                      repeat: Infinity,
                      duration: 4,
                      ease: "easeInOut",
                    }}
                    className="mb-6"
                  >
                    <svg
                      className="h-20 w-20 text-indigo-400"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2z"
                      />
                    </svg>
                  </m.div>
                  <h2 className="mb-2 text-2xl font-bold text-gray-700">
                    Добро пожаловать в Анализатор ценностей
                  </h2>
                  <p className="max-w-md text-gray-500">
                    Этот инструмент поможет вам понять, какие ценности
                    преобладают в ваших сообщениях. Вы можете ввести текст или
                    загрузить файл для анализа.
                  </p>
                </m.div>
              ) : (
                messages.map((message) => (
                  <m.div
                    key={message.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3 }}
                    className={`group relative flex ${message.isUser ? "justify-end" : "justify-start"}`}
                  >
                    {/* Кнопка копирования */}
                    {message.isUser ? (
                      <div className="max-w-3xl rounded-2xl rounded-tr-none bg-indigo-600 p-4 text-white shadow-md">
                        {message.fileName ? (
                          <div className="flex items-center">
                            <svg
                              className="mr-2 h-5 w-5"
                              fill="none"
                              stroke="currentColor"
                              viewBox="0 0 24 24"
                            >
                              <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z"
                              />
                            </svg>
                            {message.fileName}
                          </div>
                        ) : (
                          message.text
                        )}
                      </div>
                    ) : message.isLoading ? (
                      <m.div className="flex max-w-3xl items-center space-x-2 rounded-2xl rounded-tl-none bg-white p-4 shadow-md">
                        <m.div
                          animate={{ rotate: 360 }}
                          transition={{
                            repeat: Infinity,
                            duration: 1,
                            ease: "linear",
                          }}
                          className="h-5 w-5 rounded-full border-2 border-indigo-600 border-t-transparent"
                        />
                        <span>
                          {isUploading ? "Загружаем файл..." : "Анализируем..."}
                        </span>
                      </m.div>
                    ) : message.error ? (
                      <div className="max-w-3xl rounded-2xl rounded-tl-none bg-white p-4 text-red-500 shadow-md">
                        Ошибка: {message.error}
                      </div>
                    ) : (
                      <div className="w-full">
                        <div className="w-md rounded-2xl rounded-tl-none bg-white p-6 shadow-md md:w-xl">
                          <h3 className="mb-3 text-lg font-bold text-indigo-600">
                            Найденные ценности:
                          </h3>
                          <div className="space-y-3">
                            {message.categories?.map((cat, i) => (
                              <div
                                key={i}
                                className="flex flex-col sm:flex-row sm:items-center"
                              >
                                <div className="w-full font-medium sm:w-64">
                                  {cat.name}
                                </div>
                                <div className="mt-1 flex flex-1 items-center sm:mt-0">
                                  <div className="h-3 flex-1 overflow-hidden rounded-full bg-gray-200">
                                    <m.div
                                      initial={{ width: 0 }}
                                      animate={{ width: `${cat.probability}%` }}
                                      transition={{
                                        duration: 1,
                                        delay: i * 0.1,
                                      }}
                                      className="h-full bg-indigo-500"
                                    />
                                  </div>
                                  <div className="ml-2 w-12 text-right text-sm font-medium">
                                    {cat.probability.toFixed(1)}%
                                  </div>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>

                        {/* Кнопка копирования под сообщением бота */}
                        <div className="mt-2 flex justify-start">
                          <m.button
                            onClick={() =>
                              copyToClipboard(
                                formatResults(message.categories || []),
                                message.id,
                              )
                            }
                            className="rounded-lg bg-indigo-100 p-2 text-indigo-700 hover:bg-indigo-200 focus:outline-none"
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                            title="Копировать результат"
                          >
                            {copiedId === message.id ? (
                              <>
                                <svg
                                  className="h-5 w-5 text-green-500"
                                  fill="none"
                                  stroke="currentColor"
                                  viewBox="0 0 24 24"
                                >
                                  <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth={2}
                                    d="M5 13l4 4L19 7"
                                  />
                                </svg>
                              </>
                            ) : (
                              <>
                                <svg
                                  className="h-5 w-5"
                                  fill="none"
                                  stroke="currentColor"
                                  viewBox="0 0 24 24"
                                >
                                  <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth={2}
                                    d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3"
                                  />
                                </svg>
                              </>
                            )}
                          </m.button>
                        </div>
                      </div>
                    )}
                  </m.div>
                ))
              )}
              <div ref={messagesEndRef} />
            </AnimatePresence>
          </div>

          <div className="border-t border-gray-200 bg-white p-4">
            <div className="mx-auto max-w-4xl">
              <div className="relative">
                <m.textarea
                  ref={inputRef}
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Введите текст для анализа..."
                  className="w-full resize-none rounded-xl border border-gray-300 p-4 pr-32 transition-all focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200"
                  rows={3}
                  whileFocus={{
                    boxShadow: "0 0 0 3px rgba(99, 102, 241, 0.3)",
                  }}
                />

                {/* Кнопка загрузки файла */}
                <m.button
                  type="button"
                  onClick={handleUploadClick}
                  disabled={isLoading}
                  className="absolute right-16 bottom-3 rounded-lg bg-gray-200 p-2 text-gray-700 transition-colors hover:bg-gray-300 disabled:cursor-not-allowed disabled:opacity-50"
                  whileHover={{ scale: isLoading ? 1 : 1.05 }}
                  whileTap={{ scale: isLoading ? 1 : 0.95 }}
                >
                  Загрузить файл
                </m.button>

                {/* загрузка файла */}
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileChange}
                  className="hidden"
                  accept=".txt,.pdf,.doc,.docx"
                />

                {/* Кнопка отправки */}
                <m.button
                  onClick={handleSendMessage}
                  disabled={isLoading || (!inputValue && !file)}
                  className="absolute right-3 bottom-3 rounded-lg bg-indigo-600 p-2 text-white transition-colors hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50"
                  whileHover={{ scale: isLoading ? 1 : 1.1 }}
                  whileTap={{ scale: isLoading ? 1 : 0.95 }}
                >
                  {isLoading ? (
                    <m.div
                      animate={{ rotate: 360 }}
                      transition={{ repeat: Infinity, duration: 1 }}
                      className="h-5 w-5 rounded-full border-2 border-white border-t-transparent"
                    />
                  ) : (
                    <svg
                      className="h-5 w-5"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                      />
                    </svg>
                  )}
                </m.button>
              </div>

              {/* Отображение выбранного файла */}
              {file && (
                <m.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mt-2 flex items-center justify-between rounded-lg bg-indigo-50 p-2"
                >
                  <div className="flex items-center">
                    <svg
                      className="mr-2 h-5 w-5 text-indigo-600"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z"
                      />
                    </svg>
                    <span className="text-sm text-indigo-800">{file.name}</span>
                  </div>
                  <button
                    onClick={() => setFile(null)}
                    className="rounded-full p-1 text-indigo-600 hover:bg-indigo-100"
                  >
                    <svg
                      className="h-4 w-4"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M6 18L18 6M6 6l12 12"
                      />
                    </svg>
                  </button>
                </m.div>
              )}

              <p className="mt-2 text-center text-xs text-gray-500">
                Нажмите Enter для отправки, Shift+Enter для новой строки
              </p>
            </div>
          </div>
        </main>
      </div>

      {/* Правая рамка */}
      <div className="w-16 bg-indigo-50 md:w-32 lg:w-48"></div>
    </div>
  );
}
