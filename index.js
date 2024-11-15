import express, { urlencoded } from "express";
import path from "path"; // Import path module for file paths

import { fileURLToPath } from 'url';
import { dirname } from 'path';
import multer from "multer";
import { exec } from "child_process";
import fs from "fs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);



const app = express();


const upload = multer({ dest: path.join(__dirname, 'uploads') }); 

app.use(express.static(path.join(__dirname + '/public')));

const port = 3000;

app.get("/", function (req, res) {
    res.sendFile(__dirname + "/index.html");
});

app.post("/upload", upload.single('image'), function (req, res) {
    // Assuming the file is uploaded as 'image'
    const image = req.file;
    console.log("post method called");
    const mimeType = req.file.mimetype;
    console.log("MIME Type:", mimeType);

    // Call the facial recognition script with the uploaded image's path
    // const imagePath = image.path.replace(/\\/g, '/');
    // const scriptPath = path.join(__dirname, 'project.py');
    const imagePath = image.path.replace(/\\/g, '/');
   

    console.log("image path: ", imagePath);

    // Command to run the facial recognition script with the image's path
    const command = `python project.py "${imagePath}"`;
    console.log("command: ", command);
    // Execute the command
    exec(command, (error, stdout, stderr) => {
        if (error) {
            console.error(`Error executing the command: ${error}`);
            res.status(500).send('Internal Server Error');
            return;
        }
        if (stderr) {
            console.error(`stderr: ${stderr}`);
            return;
          }
          // Read the recognized names Excel file
        fs.readFile('attendance.xlsx', (err, data) => {
            if (err) {
                console.error(`Error reading recognized names file: ${err}`);
                res.status(500).send('Internal Server Error');
                return;
            }
            
            // Send the output of the script back to the client along with the Excel file
            res.set({
                'Content-Type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                'Content-Disposition': 'attachment; filename=recognized_names.xlsx'
            });
            res.send(data);
        });
    });

    // Do something with the uploaded file, such as saving it to disk or processing it

    // Send a response
    // res.send("File uploaded successfully");
});
app.listen(port, () => {
    console.log("Server is listening on port " + port);
});


