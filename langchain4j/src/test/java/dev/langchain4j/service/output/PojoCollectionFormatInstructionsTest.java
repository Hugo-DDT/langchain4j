package dev.langchain4j.service.output;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.fasterxml.jackson.core.type.TypeReference;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.model.chat.Capability;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.UserMessage;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;

class PojoCollectionFormatInstructionsTest {

    record Person(String name, int age) {}

    interface People {
        @SystemMessage("list people")
        List<Person> extract(@UserMessage String text);

        @SystemMessage("list people")
        Set<Person> extractUnique(@UserMessage String text);
    }

    static class NoFields {}

    @Test
    void list_of_pojo_appends_json_array_instructions_and_parses_the_response() {
        CapturingChatModel model =
                new CapturingChatModel("[{\"name\":\"Alice\",\"age\":30},{\"name\":\"Bob\",\"age\":25}]");
        assertThat(model.supportedCapabilities()).doesNotContain(Capability.RESPONSE_FORMAT_JSON_SCHEMA);

        People people = AiServices.builder(People.class).chatModel(model).build();

        List<Person> extracted = people.extract("Alice 30, Bob 25");

        assertThat(model.invocationCount()).isOne();
        assertThat(model.prompt())
                .contains(jsonArrayInstructions(Person.class))
                .doesNotContain("You must answer strictly in the following format:");
        assertThat(extracted).containsExactly(new Person("Alice", 30), new Person("Bob", 25));
    }

    @Test
    void set_of_pojo_reuses_the_same_json_array_instructions() {
        CapturingChatModel model =
                new CapturingChatModel("[{\"name\":\"Alice\",\"age\":30},{\"name\":\"Bob\",\"age\":25}]");
        assertThat(model.supportedCapabilities()).doesNotContain(Capability.RESPONSE_FORMAT_JSON_SCHEMA);

        People people = AiServices.builder(People.class).chatModel(model).build();

        Set<Person> extracted = people.extractUnique("Alice 30, Bob 25");

        assertThat(model.invocationCount()).isOne();
        assertThat(model.prompt()).contains(jsonArrayInstructions(Person.class));
        assertThat(extracted).containsExactlyInAnyOrder(new Person("Alice", 30), new Person("Bob", 25));
    }

    @Test
    void service_output_parser_does_not_double_prefix_collection_instructions() {
        String listInstructions =
                new ServiceOutputParser().outputFormatInstructions(new TypeReference<List<Person>>() {}.getType());
        String setInstructions =
                new ServiceOutputParser().outputFormatInstructions(new TypeReference<Set<Person>>() {}.getType());

        assertThat(listInstructions).isEqualTo(jsonArrayInstructions(Person.class));
        assertThat(setInstructions).isEqualTo(jsonArrayInstructions(Person.class));
        assertThat(listInstructions).doesNotContain("You must answer strictly in the following format:");
        assertThat(countOccurrences(listInstructions, "You must")).isOne();
    }

    @Test
    void collection_without_a_describable_element_names_the_return_type() {
        Type listOfNoFields = new TypeReference<List<NoFields>>() {}.getType();

        assertThatThrownBy(() -> new ServiceOutputParser().outputFormatInstructions(listOfNoFields))
                .isExactlyInstanceOf(IllegalStateException.class)
                .hasMessageContaining("java.util.List<" + NoFields.class.getName() + ">")
                .hasMessageContaining("JSON Schema")
                .hasMessageContaining("format instructions");
    }

    private static String jsonArrayInstructions(Class<?> elementType) {
        String elementInstructions = new PojoOutputParser<>(elementType).formatInstructions();
        String prefix = "\nYou must answer strictly in the following JSON format: ";
        assertThat(elementInstructions).startsWith(prefix);
        return prefix + "[" + elementInstructions.substring(prefix.length()) + "]";
    }

    private static int countOccurrences(String text, String fragment) {
        int count = 0;
        int from = 0;
        while (true) {
            int index = text.indexOf(fragment, from);
            if (index < 0) {
                return count;
            }
            count++;
            from = index + fragment.length();
        }
    }

    static final class CapturingChatModel implements ChatModel {

        private final String response;
        private final List<ChatRequest> requests = new ArrayList<>();

        CapturingChatModel(String response) {
            this.response = response;
        }

        @Override
        public ChatResponse doChat(ChatRequest chatRequest) {
            requests.add(chatRequest);
            return ChatResponse.builder().aiMessage(AiMessage.from(response)).build();
        }

        int invocationCount() {
            return requests.size();
        }

        String prompt() {
            StringBuilder prompt = new StringBuilder();
            for (ChatRequest request : requests) {
                for (ChatMessage message : request.messages()) {
                    prompt.append(messageText(message)).append('\n');
                }
            }
            return prompt.toString();
        }

        private static String messageText(ChatMessage message) {
            if (message instanceof dev.langchain4j.data.message.SystemMessage systemMessage) {
                return systemMessage.text();
            }
            if (message instanceof dev.langchain4j.data.message.UserMessage userMessage) {
                return userMessage.singleText();
            }
            return message.toString();
        }
    }
}
