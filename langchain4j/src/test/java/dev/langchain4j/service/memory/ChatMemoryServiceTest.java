package dev.langchain4j.service.memory;

import static org.assertj.core.api.Assertions.assertThat;

import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

class ChatMemoryServiceTest {

    @Test
    void single_memory_mode_should_return_configured_memory_for_equal_but_distinct_default_id() {
        ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);
        ChatMemoryService service = new ChatMemoryService(chatMemory);

        ChatMemory result = service.getChatMemory(new String("default"));

        assertThat(result).isSameAs(chatMemory);
    }

    @Test
    void single_memory_mode_should_return_configured_memory_for_default_constant() {
        ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);
        ChatMemoryService service = new ChatMemoryService(chatMemory);

        assertThat(service.getChatMemory(ChatMemoryService.DEFAULT)).isSameAs(chatMemory);
    }

    @Test
    void single_memory_mode_should_return_null_for_other_memory_id() {
        ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);
        ChatMemoryService service = new ChatMemoryService(chatMemory);

        assertThat(service.getChatMemory("other")).isNull();
    }

    @Test
    void invocation_chat_memory_should_forward_every_method_of_chat_memory() throws Exception {
        // a method that is not overridden would run the interface's default instead of the user's chat memory
        Class<?> invocationChatMemory = Class.forName(ChatMemoryService.class.getName() + "$InvocationChatMemory");

        List<String> notForwarded = Arrays.stream(ChatMemory.class.getMethods())
                .filter(method -> !Modifier.isStatic(method.getModifiers()))
                .filter(method -> !isDeclaredBy(invocationChatMemory, method))
                .map(Method::toGenericString)
                .toList();

        assertThat(notForwarded).isEmpty();
    }

    private static boolean isDeclaredBy(Class<?> type, Method method) {
        try {
            type.getDeclaredMethod(method.getName(), method.getParameterTypes());
            return true;
        } catch (NoSuchMethodException e) {
            return false;
        }
    }
}
