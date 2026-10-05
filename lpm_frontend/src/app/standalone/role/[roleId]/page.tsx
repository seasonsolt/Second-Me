'use client';

import { useEffect, useState, useRef } from 'react';
import { useParams } from 'next/navigation';
import ChatInput from '@/components/chat/ChatInput';
import ChatMessage from '@/components/chat/ChatMessage';
import { roleplayChatStorage, type ChatMessage as IChatMessage } from '@/utils/chatStorage';
import { getRole, type RoleRes } from '@/service/role';
import type { ChatRequest } from '@/hooks/useSSE';
import { useSSE } from '@/hooks/useSSE';
import { message } from 'antd';
import { useLoadInfoStore } from '@/store/useLoadInfoStore';

// Function to generate unique ID
const generateMessageId = () => {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
};

export default function RoleChat() {
  const params = useParams();
  const role_id = params.roleId as string;

  const [role, setRole] = useState<RoleRes | null>(null);
  const [messages, setMessages] = useState<IChatMessage[]>([]);
  const [loading, setLoading] = useState(true);
  const { sendStreamMessage, streaming, streamContent } = useSSE();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const loadInfo = useLoadInfoStore((state) => state.loadInfo);

  const scrollToBottom = () => {
    if (messagesEndRef.current) {
      const container = messagesEndRef.current.parentElement;

      container?.scrollTo({
        top: container.scrollHeight,
        behavior: 'smooth'
      });
    }
  };

  // Scroll to bottom when messages update
  useEffect(() => {
    scrollToBottom();
  }, [messages, streamContent]);

  // Load role information and chat history
  useEffect(() => {
    const loadRole = async () => {
      setLoading(true);

      try {
        const res = await getRole(role_id);

        if (res.data.code === 0) {
          setRole(res.data.data);
          // Load chat history
          const storedMessages = roleplayChatStorage.getMessages(role_id);
          const systemMessage: IChatMessage = {
            id: generateMessageId(),
            content: role?.system_prompt || '',
            role: 'system',
            timestamp: new Date().toLocaleTimeString([], {
              hour: '2-digit',
              minute: '2-digit'
            })
          };

          if (storedMessages.length === 0) {
            // If no history messages, add welcome message
            const welcomeMessage: IChatMessage = {
              id: generateMessageId(),
              content: `Hello! I am a ${res.data.data.name}. How can I help you today?`,
              role: 'assistant',
              timestamp: new Date().toLocaleTimeString([], {
                hour: '2-digit',
                minute: '2-digit'
              })
            };

            roleplayChatStorage.saveMessages(role_id, [welcomeMessage]);
            setMessages([systemMessage, welcomeMessage]);
          } else {
            setMessages([systemMessage, ...storedMessages]);
          }
        } else {
          message.error('Failed to load role');
        }
      } catch (error) {
        console.error('Failed to load role:', error);
        message.error('Failed to load role');
      } finally {
        setLoading(false);
      }
    };

    loadRole();
  }, [role_id]);

  useEffect(() => {}, [role]);

  const handleSendMessage = async (content: string) => {
    if (!role) return;

    const userMessage: IChatMessage = {
      id: generateMessageId(),
      content,
      role: 'user',
      timestamp: new Date().toLocaleTimeString([], {
        hour: '2-digit',
        minute: '2-digit'
      })
    };

    // Create an empty assistant message
    const assistantMessage: IChatMessage = {
      id: generateMessageId(),
      content: '',
      role: 'assistant',
      timestamp: new Date().toLocaleTimeString([], {
        hour: '2-digit',
        minute: '2-digit'
      })
    };

    // Update message list, add user message and empty assistant message
    let newMessages = [...messages, userMessage];

    const systemMessage: IChatMessage = {
      id: generateMessageId(),
      content: role.system_prompt || '',
      role: 'system',
      timestamp: new Date().toLocaleTimeString([], {
        hour: '2-digit',
        minute: '2-digit'
      })
    };

    if (!newMessages.find((item) => item.role === 'system')) {
      newMessages = [systemMessage, ...newMessages];
    } else {
      newMessages = newMessages.map((msg) => {
        if (msg.role === 'system') {
          return { ...msg, content: role.system_prompt || '' };
        }

        return msg;
      });
    }

    setMessages([...newMessages, assistantMessage]);
    // Save messages
    roleplayChatStorage.saveMessages(role_id, [...newMessages, assistantMessage]);

    // Send request
    const chatRequest: ChatRequest = {
      messages: newMessages.map((msg) => ({
        role: msg.role,
        content: msg.content
      })),
      metadata: {
        enable_l0_retrieval: role.enable_l0_retrieval,
        enable_l1_retrieval: role.enable_l1_retrieval ?? true,
        role_id: role.uuid
      },
      temperature: 0.01,
      stream: true
    };

    await sendStreamMessage(chatRequest);
  };

  const handleClearChat = () => {
    roleplayChatStorage.clearMessages(role_id);
    setMessages([]);
  };

  // Monitor streamContent changes to update messages
  useEffect(() => {
    if (!streamContent) return;

    setMessages((prevMessages) => {
      // Update the content of the last assistant message
      const updatedMessages = prevMessages.map((msg, index) => {
        if (index === prevMessages.length - 1 && msg.role === 'assistant') {
          return { ...msg, content: streamContent };
        }

        return msg;
      });

      // When message is complete, save the final message list
      if (!streaming) {
        roleplayChatStorage.saveMessages(role_id, updatedMessages);
      }

      return updatedMessages;
    });
  }, [streamContent, streaming, role_id]);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-50">
        <div
          className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"
          suppressHydrationWarning
        />
      </div>
    );
  }

  return (
    <div className="h-full bg-gray-50 w-full">
      <div className="flex h-full">
        {/* Left Sidebar - Role Information */}
        <div className="w-64 bg-white border-r flex flex-col">
          <div className="p-4 border-b">
            <h1 className="font-semibold text-lg truncate">{loadInfo?.name}</h1>
            <p className="text-sm text-gray-600 mt-1">as</p>
            <h2 className="font-medium text-base text-blue-600">{role?.name}</h2>
          </div>

          <div className="p-4">
            <h3 className="text-sm font-medium text-gray-700 mb-2">Role Description</h3>
            <p className="text-sm text-gray-600">
              {role?.description || 'No description available'}
            </p>
          </div>

          <div className="mt-auto p-4 border-t">
            <button
              className="w-full py-2 px-4 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded text-sm transition-colors flex items-center justify-center gap-2"
              onClick={handleClearChat}
            >
              <svg
                className="h-4 w-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                />
              </svg>
              Clear Chat
            </button>
          </div>
        </div>

        {/* Main Chat Area */}
        <div className="flex-1 flex flex-col">
          {/* Chat Messages */}
          <div className="flex-1 overflow-y-auto p-4">
            {messages.map((msg, index) => (
              <ChatMessage
                key={msg.id}
                isLoading={
                  streaming &&
                  !streamContent &&
                  index === messages.length - 1 &&
                  msg.role === 'assistant'
                }
                message={msg.content}
                role={msg.role}
                timestamp={msg.timestamp}
              />
            ))}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <ChatInput disabled={streaming} onSendMessage={handleSendMessage} />
        </div>
      </div>
    </div>
  );
}
