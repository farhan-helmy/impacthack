var amqp = require("amqplib/callback_api");
const { Server } = require("socket.io");
const io = new Server(8001, {
    cors: {
        origin: "*",
        methods: ["GET", "POST"]
    }
});

io.on("connection", (socket) => {
  console.log("a user connected");
});

amqp.connect("amqp://127.0.0.1", function (error0, connection) {
  if (error0) {
    throw error0;
  }
  connection.createChannel(function (error1, channel) {
    if (error1) {
      throw error1;
    }
    var queue = "amazonggo";

    channel.assertQueue(queue, {
      durable: false,
    });

    console.log(" [*] Waiting for messages in %s. To exit press CTRL+C", queue);

    channel.consume(
      queue,
      function (msg) {
        io.emit("stock-update", msg.content.toString());
        console.log(msg.content.toString());
      },
      {
        noAck: true,
      }
    );
  });
});
