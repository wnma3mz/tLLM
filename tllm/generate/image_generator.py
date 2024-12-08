import time
from typing import List

from tllm.img_helper import pil_image_to_base64
from tllm.schemas import ForwardResult, ImageRequestData


class ImageGenerator:
    def __init__(self, manager: "RPCManager", logger, model, tok=None) -> None:
        self.manager = manager
        self.logger = logger
        self.model = model

    def update_url(self, url: str, pp_size: int):
        self.manager.update_url(url, pp_size)

    async def forward(self, image_request: ImageRequestData) -> ForwardResult:
        s0 = time.perf_counter()
        hidden_states, text_embeddings = self.model.get_encoder_hidden_states(
            image_request.generate_iter, image_request.runtime_config, image_request.input_embeds
        )
        self.logger.debug(f"get_encoder_hidden_states cost time: {time.perf_counter() - s0:.4f}s")
        s1 = time.perf_counter()

        height, width = image_request.runtime_config.height, image_request.runtime_config.width
        seq_len = image_request.input_embeds.prompt_embeds.shape[1]

        hidden_states, calc_cost_time_list = await self.manager.image_forward(
            hidden_states, text_embeddings, seq_len, height, width, image_request.request_id
        )
        comm_cost_time = time.perf_counter() - s1 - sum(calc_cost_time_list)
        s0 = time.perf_counter()
        hidden_states = self.model.get_noise(
            image_request.generate_iter,
            image_request.runtime_config,
            hidden_states,
            image_request.input_embeds.latents,
        )
        self.logger.debug(f"get_noise cost time: {time.perf_counter() - s0:.4f}s")
        return ForwardResult(
            hidden_states=hidden_states,
            comm_cost_time=comm_cost_time,
            calc_cost_time=sum(calc_cost_time_list),
        )

    async def generate(self, request_list: List[ImageRequestData]):
        """
        @params:
        """
        # TODO: only support single sequence_request
        assert len(request_list) == 1
        # get_embedding -> [get_encoder_hidden_states -> rpc request -> get_noise] * t -> get_images
        image_request = request_list[0]

        s0 = time.perf_counter()
        if image_request.generate_iter == 0:
            image_request.start_time = time.perf_counter()
            image_request.runtime_config, image_request.input_embeds = self.model.get_embedding(
                image_request.seed, image_request.prompt, image_request.config
            )
        self.logger.debug(f"get_embedding cost time: {time.perf_counter() - s0:.4f}s")

        s0 = time.perf_counter()
        forward_result = await self.forward(image_request)
        self.logger.debug(f"decoder cost time: {time.perf_counter() - s0:.4f}s")

        image_request.generate_iter += 1
        image_request.input_embeds.latents = forward_result.hidden_states

        # TODO: async return
        s0 = time.perf_counter()
        if image_request.generate_iter == image_request.config.num_inference_steps:
            image = self.model.get_images(
                image_request.input_embeds.latents,
                image_request.runtime_config,
                image_request.seed,
                image_request.prompt,
                time.perf_counter() - image_request.start_time,
            )
            image_request.output_base64 = pil_image_to_base64(image.image)
            image.save(path="./test.png", export_json_metadata=False)
            image_request.is_stop = True

        self.logger.debug(f"get_images cost time: {time.perf_counter() - s0:.4f}s")

        fraction = forward_result.comm_cost_time / (forward_result.comm_cost_time + forward_result.calc_cost_time)
        self.logger.debug(f"communication cost time: {forward_result.comm_cost_time:.4f}s({fraction*100:.1f}%)")
        self.logger.debug("=" * 5)
