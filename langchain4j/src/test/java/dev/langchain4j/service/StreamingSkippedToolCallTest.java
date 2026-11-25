package dev.langchain4j.service;

import static dev.langchain4j.internal.InternalStreamingChatResponseHandlerUtils.onCompleteResponse;
import static dev.langchain4j.internal.InternalStreamingChatResponseHandlerUtils.onCompleteToolCall;
import static java.util.concurrent.TimeUnit.SECONDS;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import dev.langchain4j.agent.tool.Tool;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.CompleteToolCall;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutionException;
import org.junit.jupiter.api.Test;

/**
 * A streaming model can declare more tool calls on the final {@link AiMessage} than it reports through
 * {@link StreamingChatResponseHandler#onCompleteToolCall}. Those calls must not be dropped: the invocation fails
 * and the chat memory is not left with a tool call that has no result.
 */
class StreamingSkippedToolCallTest {

    interface Assistant {

        TokenStream chat(String message);
    }

    static class Tools {

        final List<String> calls = new CopyOnWriteArrayList<>();

        @Tool
        String alpha() {
            calls.add("alpha");
            return "alpha-result";
        }

        @Tool
        String beta() {
            calls.add("beta");
            return "beta-result";
        }
    }

    @Test
    void should_fail_when_a_declared_tool_call_never_completes() throws Exception {
        ToolExecutionRequest alpha = request("1", "alpha");
        ToolExecutionRequest beta = request("2", "beta");
        List<ChatRequest> requests = new CopyOnWriteArrayList<>();
        StreamingChatModel model = new StreamingChatModel() {
            @Override
            public void doChat(ChatRequest request, StreamingChatResponseHandler handler) {
                requests.add(request);
                if (requests.size() == 1) {
                    onCompleteToolCall(handler, new CompleteToolCall(0, alpha));
                    onCompleteResponse(
                            handler,
                            ChatResponse.builder()
                                    .aiMessage(AiMessage.from(List.of(alpha, beta)))
                                    .build());
                } else {
                    onCompleteResponse(
                            handler,
                            ChatResponse.builder()
                                    .aiMessage(AiMessage.from("done"))
                                    .build());
                }
            }
        };
        Tools tools = new Tools();
        ChatMemory memory = MessageWindowChatMemory.withMaxMessages(20);
        Assistant assistant = AiServices.builder(Assistant.class)
                .streamingChatModel(model)
                .tools(tools)
                .chatMemory(memory)
                .executeToolsConcurrently(Runnable::run)
                .build();

        CompletableFuture<ChatResponse> completed = new CompletableFuture<>();
        assistant
                .chat("go")
                .onPartialResponse(ignored -> {})
                .onCompleteResponse(completed::complete)
                .onError(completed::completeExceptionally)
                .start();

        assertThatThrownBy(() -> completed.get(5, SECONDS))
                .isInstanceOf(ExecutionException.class)
                .cause()
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("2")
                .hasMessageContaining("neither executed nor produced a result");

        assertThat(tools.calls).containsExactly("alpha");
        assertThat(requests).hasSize(1);
        assertEveryToolCallHasAResult(memory.messages());

        ToolExecutionResultMessage betaResult = memory.messages().stream()
                .filter(ToolExecutionResultMessage.class::isInstance)
                .map(ToolExecutionResultMessage.class::cast)
                .filter(result -> "2".equals(result.id()))
                .findFirst()
                .orElseThrow();
        assertThat(betaResult.isError()).isTrue();
        assertThat(betaResult.text()).isNotEqualTo("beta-result");
    }

    private static void assertEveryToolCallHasAResult(List<ChatMessage> messages) {
        List<String> openToolCallIds = new ArrayList<>();
        for (ChatMessage message : messages) {
            if (message instanceof AiMessage aiMessage && aiMessage.hasToolExecutionRequests()) {
                assertThat(openToolCallIds).isEmpty();
                aiMessage.toolExecutionRequests().forEach(request -> openToolCallIds.add(request.id()));
            } else if (message instanceof ToolExecutionResultMessage result) {
                assertThat(openToolCallIds).contains(result.id());
                openToolCallIds.remove(result.id());
            } else {
                assertThat(openToolCallIds).isEmpty();
            }
        }
        assertThat(openToolCallIds).isEmpty();
    }

    private static ToolExecutionRequest request(String id, String name) {
        return ToolExecutionRequest.builder().id(id).name(name).arguments("{}").build();
    }
}
