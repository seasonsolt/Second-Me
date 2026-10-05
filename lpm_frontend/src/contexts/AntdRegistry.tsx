'use client';

import { StyleProvider, createCache, extractStyle } from '@ant-design/cssinjs';
import { useServerInsertedHTML } from 'next/navigation';
import { useMemo } from 'react';
import type React from 'react';

const StyledComponentsRegistry = ({ children }: { children: React.ReactNode }): JSX.Element => {
  // One cache per mount; a new cache on every render loses registered styles and
  // crashes antd ("Cannot read properties of null (reading '1')") in production builds.
  const cache = useMemo(() => createCache(), []);

  useServerInsertedHTML(() => (
    <style dangerouslySetInnerHTML={{ __html: extractStyle(cache, true) }} id="antd" />
  ));

  return <StyleProvider cache={cache}>{children}</StyleProvider>;
};

export default StyledComponentsRegistry;
