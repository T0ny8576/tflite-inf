import logging
from gabriel_protocol import gabriel_pb2
from gabriel_server import local_engine
from gabriel_server import cognitive_engine


SOURCE = 'profiling'
INPUT_QUEUE_MAXSIZE = 60
PORT = 9099
NUM_TOKENS = 1
DETECTOR_ONES_SIZE = (1, 480, 640, 3)

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceEngine(cognitive_engine.Engine):
    def __init__(self):
        self.input_count = 0

    def handle(self, input_frame):
        status = gabriel_pb2.ResultWrapper.Status.SUCCESS
        result_wrapper = cognitive_engine.create_result_wrapper(status)

        # result = gabriel_pb2.ResultWrapper.Result()
        # result.payload_type = gabriel_pb2.PayloadType.IMAGE
        # result.payload = input_frame.payloads[0]
        # result_wrapper.results.append(result)
        self.input_count += 1
        logger.info("Input count: {}".format(self.input_count))

        return result_wrapper


def main():
    def engine_factory():
        return InferenceEngine()

    local_engine.run(engine_factory, SOURCE, INPUT_QUEUE_MAXSIZE,
                     PORT, NUM_TOKENS)


if __name__ == '__main__':
    main()
