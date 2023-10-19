$(document).ready(function () {
    let refresh_progress = function () {
        let status = $('#status').val()
        if (status === 'done' || status === 'error') {
            return;
        }
        $.get("/progress",
            {request_id: $('#request_id').val()},
            function (data, status) {
                if (status === 'success') {
                    $('#search-result-step').html(data.html);
                    $('#result-text')[0].innerText = data.openai_stream;
                }
            }
        );
    }

    let submit_search = function (is_poll, event) {
        if (event) {
            event.preventDefault();
        }
        let search_text = $('#form1').val();
        $('#search-btn')[0].disabled = true;
        $('#status').val('processing');
        $('#search-result-spinner').addClass('d-flex');
        // $('#search-results').hide();
        $('#search_text')[0].innerText = search_text;
        $('#search_result_sources')[0].innerText = '';
        $('#explain_results').hide();
        $.ajax({
            url: '/search',
            type: 'POST',
            data: {
                q: search_text,
                request_id: $('#request_id').val(),
                bing_search_subscription_key: $('#bing_search_subscription_key').val(),
                openai_api_key: $('#openai_api_key').val(),
                is_use_source: $('input[name="is_use_source"]')[0].checked,
                llm_service_provider: $('#llm_service_provider').val(),
                llm_model: $('#llm_model').val(),
                language: $('#language').val()
            },
            success: function (response) {
                $('#' + response.id).html(response.html)
                $('#explain_results').html(response.explain_html)
                $('#request_id_status_html').html(response.request_id_status_html)
                $('#search-btn')[0].disabled = false;
                $('#search-result-spinner').removeClass('d-flex');
                $('#search-results').show();
                $('#explain_results').show();
            },
            error: function (error) {
                console.log(error)
                $('#explain_results').html(response.explain_html)
                $('#request_id_status_html').html(response.request_id_status_html)
                $('#search-btn')[0].disabled = false;
                $('#search-result-spinner').removeClass('d-flex');
                $('#search-results').show();
                $('#explain_results').show();
            }
        })

        // call 10 times progress each sec
        if (is_poll) {
            CALL_TIMES = 15; // 2 sec for 30 sec
            for (let i = 1; i < CALL_TIMES + 1; i++) {
                setTimeout(refresh_progress, 2000 * i);
            }
        }

    }

    $('.prompt-ex-btn').click(function () {
        $('#form1').val($(this).text())
        submit_search(false, null);
    });

    $('form').submit(function (event) {
        submit_search(true, event);
    })
})

/*
这段代码是一个 JavaScript 脚本，它使用 jQuery 库来操作 HTML 页面中的元素。下面是对每一行代码的注释：

$(document).ready(function () {：当文档加载完成后，执行此函数。
let refresh_progress = function () {：定义一个名为 refresh_progress 的函数。
let status = $('#status').val()：获取 HTML 元素中 ID 为 status 的元素的值，并将其赋值给 status 变量。
if (status === 'done' || status === 'error') {：如果 status 变量的值为 'done' 或 'error'，则执行以下操作：
return;：退出函数。
}：结束 if 语句。
$.get("/progress",：使用 jQuery 的 get 方法向 /progress URL 发送一个 HTTP GET 请求。
{request_id: $('#request_id').val()},：将 ID 为 request_id 的 HTML 元素的值作为请求参数发送。
function (data, status) {：定义一个回调函数，用于处理服务器返回的响应。
if (status === 'success') {：如果响应状态为 'success'，则执行以下操作：
$('#search-result-step').html(data.html);：将服务器返回的数据中的 html 属性的值设置为 ID 为 search-result-step 的 HTML 元素的内容。
$('#result-text')[0].innerText = data.openai_stream;：将服务器返回的数据中的 openai_stream 属性的值设置为 ID 为 result-text 的 HTML 元素的文本内容。
}：结束 if 语句。
}：结束回调函数定义。
);：结束 jQuery 的 get 方法调用。
}：结束 refresh_progress 函数定义。
17-19: 定义一个名为 submit_search 的函数，该函数接受两个参数：is_poll 和 event。

20: 如果存在事件对象，则阻止其默认行为。

21: 获取 ID 为 'form1' 的 HTML 元素的值，并将其赋值给变量 search_text.

22: 禁用 ID 为 'search-btn' 的 HTML 元素。

23: 将 ID 为 'status' 的 HTML 元素的值设置为 'processing'.

24: 为 ID 为 'search-result-spinner' 的 HTML 元素添加类 'd-flex'.

25-32: 设置一些 HTML 元素的内容和属性。

33-44: 使用 jQuery 的 ajax 方法向 /search URL 发送一个 HTTP POST 请求，并传递一些参数。

45-53: 定义一个回调函数，用于处理服务器返回的响应。

54-62: 定义一个回调函数，用于处理服务器返回的错误。

63-67: 如果 is_poll 参数为 true，则多次调用 refresh_progress 函数。

68-70: 当类名为 'prompt-ex-btn' 的 HTML 元素被点击时，执行以下操作：

71: 将 ID 为 'form1' 的 HTML 元素的值设置为被点击元素的文本内容。

72: 调用 submit_search 函数，并传递 false 和 null 作为参数。

73-75: 当表单被提交时，执行以下操作：

76: 调用 submit_search 函数，并传递 true 和 event 对象作为参数。

77: }): 结束文档加载完成后执行的函数。

这段代码定义了两个函数：refresh_progress 和 submit_search。refresh_progress 函数用于向服务器发送 HTTP GET 请求以获取搜索进度信息。submit_search 函数用于向服务器发送 HTTP POST 请求以提交搜索表单，并处理服务器返回的响应。此外，这段代码还定义了一些事件处理程序，用于在用户与页面交互时触发相应的操作。希望这些注释能够帮助您理解这段代码！如果您有其他问题，请随时告诉我。
*/