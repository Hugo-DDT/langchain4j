package dev.langchain4j.service;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import dev.langchain4j.model.chat.Capability;
import dev.langchain4j.model.chat.mock.ChatModelMock;
import dev.langchain4j.service.output.OutputParsingException;
import java.util.Map;
import org.junit.jupiter.api.Test;

/**
 * {@code Map} return types are parsed as JSON. The prompt sent to the model must say so.
 */
class AiServicesMapReturnTypeTest {

    interface MetadataExtractor {

        Map<String, Object> meta(@UserMessage String text);
    }

    @Test
    void should_send_json_instructions_and_parse_json_object() {
        ChatModelMock model = ChatModelMock.thatAlwaysResponds("{\"capital\":\"Paris\"}");

        MetadataExtractor extractor =
                AiServices.builder(MetadataExtractor.class).chatModel(model).build();

        Map<String, Object> meta = extractor.meta("capital of France?");

        assertThat(model.userMessageText())
                .as("the model must be told to answer with JSON, because the parser only accepts JSON")
                .isEqualTo("capital of France?\nYou must answer strictly in the following format: JSON object");
        assertThat(meta).containsEntry("capital", "Paris");
    }

    @Test
    void should_send_json_instructions_when_model_supports_json_schema() {
        // Map has no JSON schema, so a schema-capable model still receives the textual instruction.
        ChatModelMock model = ChatModelMock.thatAlwaysResponds("{\"capital\":\"Paris\"}")
                .withSupportedCapabilities(Capability.RESPONSE_FORMAT_JSON_SCHEMA);

        MetadataExtractor extractor =
                AiServices.builder(MetadataExtractor.class).chatModel(model).build();

        Map<String, Object> meta = extractor.meta("capital of France?");

        assertThat(model.requests().get(0).responseFormat()).isNull();
        assertThat(model.userMessageText())
                .isEqualTo("capital of France?\nYou must answer strictly in the following format: JSON object");
        assertThat(meta).containsEntry("capital", "Paris");
    }

    @Test
    void should_reject_non_json_text_when_return_type_is_map() {
        ChatModelMock model = ChatModelMock.thatAlwaysResponds("Paris");

        MetadataExtractor extractor =
                AiServices.builder(MetadataExtractor.class).chatModel(model).build();

        assertThatThrownBy(() -> extractor.meta("capital of France?")).isInstanceOf(OutputParsingException.class);
        assertThat(model.userMessageText())
                .as("a non-JSON answer is rejected, so the outgoing prompt must have requested JSON")
                .isEqualTo("capital of France?\nYou must answer strictly in the following format: JSON object");
    }
}
