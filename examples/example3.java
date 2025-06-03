private void loadImage() {
        String filename = filenames[index];
        try {
            if ((filesRead % SAMPLE_SIZE) == 0) {
                fileStartTime = System.currentTimeMillis();
            }
            this.currentImage = ImageIO.read(new File(filename));
            if (currentImage == null) {
                System.out.println("Error: " + filename + " - couldn't read!");
                return;
            }
            ++filesRead;
            if ((filesRead % SAMPLE_SIZE) == 0) {
                long endTime = System.currentTimeMillis();
                float msec = (float) (endTime - fileStartTime) / SAMPLE_SIZE;
                System.out.println("Average load time = " + msec + " milliseconds");
            }
            if (displaying) {
                label.setIcon(new ImageIcon(currentImage));
                frame.setTitle("Image I/O Flipper - " + filename);
                imagePanel.repaint();
            }
        } catch (Exception exc) {
            System.out.println("\nError: " + filename + " - exception during read!");
            exc.printStackTrace();
            System.out.println();
        }
    }