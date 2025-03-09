# Setup

The simplest way to set up JGNN is to download it as a JAR package [release](https://github.com/MKLab-ITI/JGNN/releases) and add it your Java project's build path. Those working with Maven or Gradle can instead add JGNN's latest nightly release as a dependency from its JitPack distribution. Follow the link below, and click on "get it" on a particular version for full instructions. If you are the first person using the release, you might need to wait a little (less than a minute) until Jitpack finishes packaging it for everybody.

[![download JGNN](https://jitpack.io/v/MKLab-ITI/JGNN.svg)](https://jitpack.io/#MKLab-ITI/JGNN)

For example, the fields in the snippet below may be added in a Maven `pom.xml` file to work with the latest nightly release. Replace `SNAPSHOT` with the release name found in the button above.

```xml
<repositories>
    <repository>
        <id>jitpack.io</id>
        <url>https://jitpack.io</url>
    </repository>
</repositories>
<dependencies>
    <dependency>
        <groupId>com.github.MKLab-ITI</groupId>
        <artifactId>JGNN</artifactId>
        <version>SNAPSHOT</version>
    </dependency>
</dependencies>
```