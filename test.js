import { exec } from "child_process";

// Define the Python script command with arguments
const imagePath = "C:/Users/samee.WINDOWS-2L0L316/OneDrive/Desktop/DL project/uploads/4f0006cd34362874e4cebf487c2ab0ca"
const pythonScript = `python project.py "${imagePath}"`;

// Execute the Python script
exec(pythonScript, (error, stdout, stderr) => {
  if (error) {
    console.error(`Error: ${error.message}`);
    return;
  }
  if (stderr) {
    console.error(`stderr: ${stderr}`);
    return;
  }
  console.log(`stdout: ${stdout}`);
});
